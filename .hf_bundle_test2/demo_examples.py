"""
Demo showing exact output format for PromptOptEnv.
Run: python demo_examples.py
"""
import os
import sys

# Set dummy token for demo
os.environ.setdefault("HF_TOKEN", "dummy")

from optimize import PromptOptimizer

def demo_summarization():
    """Demo 1: Summarization with quality improvement"""
    print("\n" + "="*70)
    print("TASK 1: SUMMARIZATION")
    print("="*70)

    opt = PromptOptimizer(
        task_type="Summarization",
        initial_prompt="Summarize this text.",
        input_text="Artificial intelligence is transforming industries by enabling automation, improving decision-making, and creating new opportunities across sectors.",
        reference="AI is transforming industries through automation, enhanced decision-making, and new opportunities.",
        example="Key impacts: automation, decision-making, opportunities.",
        constraint="Provide a concise summary highlighting key points."
    )
    result = opt.optimize(max_steps=8)

    return result

def demo_summarization_tradeoff():
    """Demo 2: Summarization showing cost-awareness trade-off"""
    print("\n" + "="*70)
    print("TASK 1: SUMMARIZATION (Sample 2 - Cost-Aware Trade-off)")
    print("="*70)
    print("\n[+GOLD] Sample 2 (showing cost-awareness)")

    opt = PromptOptimizer(
        task_type="Summarization",
        initial_prompt="Summarize the following paragraph in detail with all possible points included.",
        input_text="Artificial intelligence is transforming industries by enabling automation, improving decision-making, and creating new opportunities across sectors.",
        reference="AI is transforming industries through automation, enhanced decision-making, and new opportunities.",
        constraint="Summarize the key points concisely.",
        example="AI transforms industries via automation, decisions, opportunities."
    )
    result = opt.optimize(max_steps=8)

    return result

def demo_qa():
    """Demo 3: Question Answering"""
    print("\n" + "="*70)
    print("TASK 2: QUESTION ANSWERING")
    print("="*70)

    opt = PromptOptimizer(
        task_type="Question Answering",
        initial_prompt="Answer the question.",
        question="What is the capital of France?",
        reference="The capital of France is Paris.",
        example="The answer is [specific fact].",
        constraint="Provide a precise and factual answer."
    )
    result = opt.optimize(max_steps=8)

    return result

def demo_qa_context():
    """Demo 4: QA with context-aware improvement"""
    print("\n" + "="*70)
    print("TASK 2: QUESTION ANSWERING (Sample 2 - Context-Aware)")
    print("="*70)
    print("\n[+] Sample 2 (context-aware improvement)")

    opt = PromptOptimizer(
        task_type="Question Answering",
        initial_prompt="Answer the question.",
        context="Photosynthesis is the process by which plants convert sunlight into energy.",
        question="What is photosynthesis?",
        reference="Photosynthesis is the process by which plants convert sunlight into energy.",
        example="It is the process where [subject] [action].",
        constraint="Answer using the provided context with clarity and accuracy."
    )
    result = opt.optimize(max_steps=8)

    return result

def main():
    print("\n" + ">>" * 35)
    print("   PROMPT OPTIMIZER - Demo Examples")
    print(">>" * 35)
    print("\n   Showing before/after with metrics")
    print("   Token cost analysis included\n")

    # Run all demos
    demo_summarization()
    input("\n> Press Enter to continue...")

    demo_summarization_tradeoff()
    input("\n> Press Enter to continue...")

    demo_qa()
    input("\n> Press Enter to continue...")

    demo_qa_context()

    print("\n" + "="*70)
    print("[OK] All demos completed!")
    print("="*70)
    print("\n[i] To optimize your own prompts, run:")
    print("   python optimize.py")
    print("\n[i] To run inference baseline, run:")
    print("   python inference.py")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
