Okay, here is a README focusing on explaining your approach for the assignment, written in normal language and suitable for accompanying the Jupyter Notebook.


# Assignment Solution: Hybrid Memory Retrieval for AI Assistants

This Jupyter Notebook is my solution to the assignment focusing on retrieving user-specific memories that might be indirectly related to a user's input, going beyond simple keyword matching or direct semantic similarity.

## The Problem I'm Addressing

The assignment posed a key challenge for AI assistants, something I have been personally working on

👉 A user asks something seemingly simple, like "Any ideas for a holiday?"

An assistant using only basic search might look for "holiday," "vacation," or synonyms. However, truly "intelligent" assistance requires understanding the user's personal context. This context often exists in scattered memories that aren't directly mentioned in the query. Examples from the assignment include:

- "I’m scared of flying."
- "I don’t like group trips."
- "My dog’s name is Arlo."

These memories are crucial for planning a relevant holiday suggestion (e.g., suggesting a road trip instead of a flight, considering pet care needs), but they have no direct semantic link to the word "holiday." The goal is to retrieve these *indirectly* relevant facts.

## My Approach: A Hybrid Strategy

To solve the problem of retrieving *indirectly* related memories, I've used a hybrid approach combining structured knowledge, semantic understanding, and LLM reasoning. Relying solely on semantic similarity (finding text that *means* similar things) isn't enough, because indirect relationships are about *connections* and *implications*, not just textual similarity.

My strategy involves these steps, all encapsulated within the main retrieval function:

1.  **Structured Memory Storage:** I store memories not just as plain text, but as **Nodes** in a basic **Knowledge Graph (KG)** using SQLite. Nodes represent concepts or facts (like "Dog Arlo", "Lactose Intolerance", "Japan Interest").
2.  **Relationship Mapping:** Crucially, I define **Relationships** between these nodes (like "Dog Arlo" --`cared_by`--> "Sister Pet Sitting", or "Recent Ear Infection" --`leads_to`--> "Diving Restriction"). These relationships are the key to capturing *indirect* links that semantic search misses.
3.  **Semantic Embedding:** The text content of each Node is also converted into a numerical representation (an **embedding**) using a powerful embedding model (from OpenAI). These embeddings are stored in a **Vector Database (ChromaDB)**, allowing for efficient semantic search.
4.  **Multi-Step Retrieval Flow:** When a user provides input:
    *   **Initial Semantic Hook:** The system first performs a semantic search using the vector database to find memories that are *textually or semantically similar* to the query. These act as potential starting points in the knowledge graph.
    *   **Graph Exploration:** From the initial semantically matched nodes, the system traverses the knowledge graph via the defined relationships. This exploration discovers nodes that are *connected* to the initial matches, even if their text isn't semantically similar to the original query (e.g., query about "holiday" might hit "Scuba Hobby" semantically, then the graph traversal finds "Recent Ear Infection" because "Scuba Hobby" --`constrained_by`--> "Recent Ear Infection").
    *   **LLM Filtering and Prioritization:** The nodes found through *both* semantic search and graph exploration are collected. This pool of candidates is then sent to a Large Language Model (LLM, via OpenRouter) along with the original user query. The LLM's task is to act as a high-level filter, evaluating which of these candidates are *most relevant* in the context of the user's query and the potential implications of the memory. This step helps prune irrelevant nodes brought in by broad graph traversal or weak semantic matches.
5.  **Return Relevant Memories:** The function returns the text content of the nodes selected by the LLM as the final list of relevant memories.

This hybrid approach allows the system to understand that a query about "holiday" should potentially trigger memories about fears (flying), responsibilities (pet care), health restrictions (ear infection affecting diving), and preferences (budget, hotel type), not just because those memories might contain related words, but because they are *linked* in the user's overall context graph.

## Key Components Used

*   **SQLite:** Stores the structured knowledge graph (Nodes and Relationships).
*   **ChromaDB:** Stores vector embeddings of memory text for semantic search.
*   **OpenAI Embeddings:** Generates the embeddings for text content.
*   **LLM via OpenRouter:** Provides the reasoning layer to filter and prioritize candidate memories found through search and graph traversal.

## The `retrieve_relevant_memories` Function

The core logic is implemented within the single function as required:

```python
def retrieve_relevant_memories(user_input: str) -> List[str]:
    # ... implementation combining semantic search,
    #     graph exploration, and LLM filtering ...
    pass # Returns a list of strings (the text of relevant memories)
```

This function encapsulates the entire hybrid process described above. It takes the user's query string as input and outputs a list of strings, where each string is the text of a memory deemed relevant by the system's multi-step retrieval strategy.

## The Memory Dataset

The notebook includes a section that **generates a synthetic dataset** of user memories. This dataset is structured to explicitly include:

*   Facts with direct semantic relevance to common topics (e.g., "Scuba Hobby" related to "diving").
*   Facts linked via relationships that represent indirect but important connections (e.g., "Dog Arlo" connected to "Sister Pet Sitting" which is connected to "Chloe Trip").
*   Facts covering diverse topics (pets, travel, health, diet, professional life) to test the system's ability to find relevant information across different domains and ignore irrelevant ones.

The creation of this dataset is shown in the notebook, demonstrating how nodes and relationships are added to build the graph structure necessary for indirect retrieval. The size and variety of this dataset are intended to clearly show how the hybrid approach avoids retrieving false positives based solely on weak semantic links, while correctly identifying truly relevant, indirectly connected facts.

## Running the Notebook

1.  Ensure you have Python and JupyterLab/Notebook installed.
2.  Install the required libraries (`pip install numpy requests openai chromadb python-dotenv jupyterlab`).
3.  Set your `OPENAI_API_KEY` and `OPENROUTER_API_KEY` environment variables, or create a `.env` file in the notebook's directory.
4.  Launch JupyterLab/Notebook and open the `.ipynb` file.
5.  Run the cells sequentially from top to bottom.

The notebook's output will show the initialization steps, the memory setup process, and then demonstrate the `retrieve_relevant_memories` function in action for several example queries, including specific tests designed to check how it handles queries where no direct information exists (hallucination tests) versus queries about existing facts (true fact tests).

## Scalability

While this implementation uses lightweight local databases (SQLite and ChromaDB persistent client), the underlying *approach* of combining semantic search (vector databases scale well) with graph traversal (dedicated graph databases like Neo4j scale much better than SQLite for complex graphs) and LLM filtering (managed by controlling the number of candidates sent) is designed with scalability in mind for a more production-ready system. The primary LLM call is on a limited set of candidates, reducing latency compared to sending the entire memory database to the LLM.

## Conclusion

By implementing a hybrid retrieval system that layers graph-based relationship understanding on top of semantic search and uses an LLM for final relevance filtering, this notebook demonstrates an effective way to retrieve user memories that are not just textually similar, but are truly relevant through indirect connections, addressing the core challenge of the assignment.
