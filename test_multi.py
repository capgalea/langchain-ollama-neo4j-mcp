from main_multi import MultiToolAgent, MCP_SERVER_CONFIGS
import asyncio

EVALUATIONS = [
    {
        "question": "How many nodes are in the graph?",
        "expected_answer": "28"
    },
    {
        "question": "How many Products are there?",
        "expected_answer": "7"
    },
    {
        "question": "How many Employees work for Acme Inc?",
        "expected_answer": "12"
    },
    # {
    #     "question": "Which department does Alice Brown work for?",
    #     "expected_answer": "Finance"
    # },
]

TEST_CONFIG = {
    "models": ["llama3.2", "mistral", "qwen3"],
    "iterations": 3
}

async def calculate_averages(config: dict, evaluations: list[dict]) -> dict:
    """
    Calculate performance metrics for model evaluations.
    
    Args:
        config: Configuration containing 'models' and 'iterations'
        evaluations: List of evaluation dictionaries with 'question' and 'expected_answer'
        
    Returns:
        dict: Results with metrics for each model
    """
    
    print("\nRunning evaluations...")
    
    results = []
    
    for model in config["models"]:

        print(f"\nTesting model: {model}...")

        # Initialize the agent
        agent = MultiToolAgent(model, MCP_SERVER_CONFIGS)
        await agent.initialize()

        model_results = {
            "model": model,
            "iterations_ran": 0,
            "total_seconds": 0.0,
            "success_rates": {
                # question: success_rate
            },
            "total_runs": 0
        }
        
        for evaluation in evaluations:
            question = evaluation["question"]
            expected_answer = evaluation["expected_answer"]
            correct_answers = 0
            total_attempts = 0
            
            print(f"\nRunning evaluation for {model} on question: {question}")
            print(f"Expected answer: {expected_answer}")
            
            # Run the evaluation for the specified number of iterations
            for i in range(config["iterations"]):
                try:
                    # Run the agent
                    result = await agent.run_request(question, with_logging=False)
                    
                    # Get the agent's answer and time taken
                    answer = str(result.get("answer", "")).strip()
                    time_taken = result.get("seconds_to_complete", 0)
                    
                    # Update metrics
                    model_results["iterations_ran"] += 1
                    model_results["total_seconds"] += time_taken
                    total_attempts += 1
                    
                    # Check if the answer is correct (case-insensitive partial match)
                    is_correct = expected_answer.lower() in answer.lower()
                    if is_correct:
                        correct_answers += 1
                    
                    print(f"  Attempt {i+1}: {answer} (Time: {time_taken:.2f}s) - " + 
                         ("✓" if is_correct else "✗"))
                    
                except Exception as e:
                    print(f"  Error in attempt {i+1}: {str(e)}")
            
            # Calculate success rate for this question
            question_success_rate = (correct_answers / total_attempts * 100) if total_attempts > 0 else 0
            model_results["success_rates"][question] = round(question_success_rate, 2)
            
            print(f"  Success rate for this question: {question_success_rate:.1f}%")
        
        # Calculate average time per question
        avg_seconds = (model_results["total_seconds"] / model_results["iterations_ran"]) if model_results["iterations_ran"] > 0 else 0
        
        # Calculate overall success rate
        if model_results["success_rates"]:
            overall_success_rate = sum(model_results["success_rates"].values()) / len(model_results["success_rates"])
        else:
            overall_success_rate = 0
        
        results.append({
            "model": model,
            "iterations_ran": model_results["iterations_ran"],
            "avg_seconds_to_complete": round(avg_seconds, 2),
            "overall_success_rate": round(overall_success_rate, 2),
            "success_rates": model_results["success_rates"]
        })

        print(f"\nFinished processing Model: {model}")
    
    return results
 
async def main():
    print("\nRunning simple evaluations...")
    results = await calculate_averages(TEST_CONFIG, EVALUATIONS)
    print("\nEvaluation Results:")
    for result in results:
        print(f"Model: {result['model']}")
        print(f"Iterations Ran: {result['iterations_ran']}")
        print(f"Average Seconds to Complete: {result['avg_seconds_to_complete']}")
        print(f"Overall Success Rate: {result['overall_success_rate']}%")
        print("Success Rates:")
        for question, success_rate in result['success_rates'].items():
            print(f"  {question}: {success_rate}%")
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())