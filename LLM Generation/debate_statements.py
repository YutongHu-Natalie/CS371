import json
from concurrent.futures import ThreadPoolExecutor
import logging
import time
from anthropic import Anthropic
from conda.exports import root_dir
from openai import OpenAI
import os
from dotenv import load_dotenv
from tqdm import tqdm
from datetime import datetime
import copy

# Load environment variables
load_dotenv()
CLAUDE_KEY = os.getenv('Claude_key')
GPT_KEY = os.getenv('gpt_key')

# Define valid topics as a set
VALID_TOPICS = {
    'Economy', 'Healthcare', 'Education', 'Immigration',
    'Climate', 'Foreign Policy', 'Social Justice', 'Infrastructure',
    'Gun Control', 'Government Policy', 'Veterans Affairs', 'Technology'
}

ANALYSIS_PROMPT = '''You are a Political analyst analyzing this presidential debate statement. 
Analyze the following statement and provide ONLY a comma-separated list of values in this exact format:
[topics],[importance],[classification],[ideology]

Example output format: "Economy/Immigration,1,2,8" or "Social Justice,0,1,-1"

Use these criteria:

TOPICS (ONLY use these exact terms, separate multiple with /):
Economy, Healthcare, Education, Immigration, Climate, Foreign Policy, Social Justice, Infrastructure, Gun Control, Government Policy, Veterans Affairs, Technology
Only include topics that are directly discussed in the statement.

IMPORTANCE:
1: Contains meaningful political content
0: Lacks meaningful political content

CLASSIFICATION:
0: Not verifiable
1: Verifiable by fact-checking
2: Verifiable by contradiction
3: Verifiable by both methods

IDEOLOGY SCORE:
0: Extremely left-wing
5: Neutral
10: Extremely right-wing
-1: No ideological content

Statement to analyze: '''


def setup_logging():
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    file_handler = logging.FileHandler(f'logs/analysis_log_{timestamp}.txt')
    console_handler = logging.StreamHandler()

    file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_format = logging.Formatter('%(levelname)s: %(message)s')
    file_handler.setFormatter(file_format)
    console_handler.setFormatter(console_format)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def process_statement(statement, model="claude", max_retries=5):
    content = statement["Statement Content"]
    statement_id = statement["Statement ID"]
    prompt = ANALYSIS_PROMPT + content

    for attempt in range(max_retries):
        try:
            if model == "claude":
                client = Anthropic(api_key=CLAUDE_KEY)
                response = client.messages.create(
                    model="claude-3-5-haiku-latest",
                    max_tokens=150,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.6
                )
                result = response.content[0].text
            else:  # gpt-4
                client = OpenAI(api_key=GPT_KEY)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,
                    temperature=0.6
                )
                result = response.choices[0].message.content

            # Parse response and update statement
            topics_str, importance, classification, ideology = result.strip().split(',')

            # Convert topics string to set, clean and validate topics
            topics_set = {topic.strip() for topic in topics_str.split('/')}
            # Only keep valid topics
            valid_topics = topics_set.intersection(VALID_TOPICS)

            # Update the fields
            statement["Statement Topic"] = list(valid_topics)
            statement["Statement Importance"] = importance
            statement["Statement Classification"] = classification
            statement["Statement Ideology Score"] = ideology
            statement["Processed"] = True

            logging.info(f"""
                Successfully processed statement {statement_id}:
                Topics: {valid_topics}
                Importance: {importance}
                Classification: {classification}
                Ideology: {ideology}
            """)

            return statement, None

        except Exception as e:
            error_msg = f"Attempt {attempt + 1} failed for statement {statement_id}: {str(e)}"
            logging.error(error_msg)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                statement["Processed"] = False
                return statement, error_msg


def process_statements_batch(statements, model="claude", max_workers=5, pbar=None):
    failed_statements = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for stmt in statements:
            future = executor.submit(process_statement, stmt, model)
            future.add_done_callback(lambda p: pbar.update() if pbar else None)
            futures.append(future)

        results = []
        for future in futures:
            statement, error = future.result()
            results.append(statement)
            if error:  # If there was an error, add to failed statements
                failed_statements.append({
                    'statement_id': statement['Statement ID'],
                    'error': error
                })

    return results, failed_statements


def process_model_statements(data, model, output_file, failed_file):
    # Create a deep copy for this model's processing
    model_data = copy.deepcopy(data)["statements"]
    batch_size = 10
    total_batches = (len(model_data) + batch_size - 1) // batch_size
    all_failed_statements = []

    # Create main progress bar for all statements
    with tqdm(total=len(model_data),
              desc=f"{model.upper()} Processing",
              position=0,
              leave=True) as pbar:

        # Create progress bar for batches
        with tqdm(total=total_batches,
                  desc="Batches",
                  position=1,
                  leave=True) as batch_pbar:

            for i in range(0, len(model_data), batch_size):
                batch = model_data[i:i + batch_size]
                processed_batch, failed_statements = process_statements_batch(
                    batch,
                    model=model,
                    max_workers=5,
                    pbar=pbar
                )

                model_data[i:i + batch_size] = processed_batch
                all_failed_statements.extend(failed_statements)

                # Save intermediate results
                with open(output_file, 'w') as f:
                    json.dump(model_data, f, indent=2)

                # Save failed statements
                if failed_statements:
                    with open(failed_file, 'w') as f:
                        json.dump(all_failed_statements, f, indent=2)

                batch_pbar.update(1)
                time.sleep(2)

    return model_data, all_failed_statements


def analyze_topics(processed_data):
    # Get all unique topics
    all_topics = set()
    for statement in processed_data:
        all_topics.update(statement["Statement Topic"])

    # Count frequency of each topic
    topic_counts = {topic: 0 for topic in VALID_TOPICS}
    for statement in processed_data:
        for topic in statement["Statement Topic"]:
            topic_counts[topic] += 1

    return topic_counts


def main():
    candidate = "harris"
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    model = "gpt"
    # Define output files for this model
    output_file = os.path.join(root_dir, "Content", "LLM outputs", "Debate Statements",
                               f'{model}_processed_{candidate}_statements.json')
    failed_file = os.path.join(root_dir, "Content", "LLM outputs", "Debate Statements",
                               f'{model}_failed_{candidate}_statements.json')
    input_file = os.path.join(root_dir, "Content", "Processed Materials", f"{candidate}_debate",
                              f'{candidate}_debate_statements.json')

    logger = setup_logging()
    start_time = time.time()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    try:
        # Verify API keys
        if not CLAUDE_KEY or not GPT_KEY:
            raise ValueError("API keys not found in .env file")

        # Load JSON file
        with open(input_file, 'r') as f:
            original_data = json.load(f)

        # Process with each model sequentially
        model_start_time = time.time()

        logger.info(f"\nStarting {model.upper()} processing...")
        processed_data, failed_statements = process_model_statements(
            original_data,
            model,
            output_file,
            failed_file
        )

        # Analyze topics for this model
        topic_counts = analyze_topics(processed_data)
        logger.info(f"\nTopic distribution for {model.upper()}:")
        for topic, count in topic_counts.items():
            logger.info(f"{topic}: {count}")

        # Log failed statements summary
        if failed_statements:
            logger.error(f"""
                    {model.upper()} Processing - Failed Statements Summary:
                    Total Failed: {len(failed_statements)}
                    Failed IDs: {[f['statement_id'] for f in failed_statements]}
                """)

        model_time = time.time() - model_start_time
        logger.info(f"""
                Completed {model.upper()} processing:
                - Processing time: {model_time:.2f}s
                - Average time per statement: {model_time / len(original_data):.2f}s
                - Failed statements: {len(failed_statements)}
                - Results saved to: {output_file}
                """)

        total_time = time.time() - start_time
        logger.info(f"""
                Analysis completed:
                - Total processing time: {total_time:.2f}s
                - Average time per statement: {total_time / (len(original_data) * 2):.2f}s
                 """)

    except Exception as e:
        logger.error(f"Critical error in main process: {str(e)}", exc_info=True)
    raise

if __name__ == "__main__":
    main()
