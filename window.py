from semaforce.window_analyzer import SemanticWindowAnalyzer


def main():
    text = """
    Artificial intelligence has made significant strides in recent years. 
    Machine learning algorithms can now perform tasks that were once thought to be the exclusive 
    domain of human intelligence. 
    This progress has led to breakthroughs in various fields, from healthcare to finance.
    However, the rapid advancement of AI also raises ethical concerns. 
    Questions about privacy, job displacement, and the potential for AI to be used maliciously 
    are at the forefront of public discourse. 
    Policymakers and technologists alike are grappling with these issues.
    Despite these challenges, the potential benefits of AI are immense. 
    From improving medical diagnoses to optimizing energy consumption, AI has the power to solve 
    some of humanity's most pressing problems. 
    As we continue to develop these technologies, it's crucial that we do so responsibly and with 
    careful consideration of their implications.
    """

    # with open("data.txt", "r") as file:
    #     text = file.read()

    analyzer = SemanticWindowAnalyzer()
    optimal_window, score, all_scores = analyzer.analyze_window_sizes(text)
    if optimal_window is not None:
        print(f"Optimal window size: {optimal_window}")
        print(f"Best score: {score}")
        print("All scores:", all_scores)
    else:
        print("Could not determine an optimal window size")


if __name__ == "__main__":
    main()
