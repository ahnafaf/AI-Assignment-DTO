

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
3.  **Semantic Embedding:** The text content of each Node is also converted into a numerical representation (an **embedding**) using `text-embedding-3-large` which is OpenAIs 3rd gen embedding model. These embeddings are stored in a **Vector Database (ChromaDB)**.
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

## Testing Set Output
✓ Database already contains 20 nodes. Skipping setup.

================================================================================
 STANDARD EXAMPLES
================================================================================

======================================================================
 EXAMPLE 1: Any ideas for a holiday in July?
======================================================================

Processing query: 'Any ideas for a holiday in July?'

Step 1: Performing initial semantic search
Found semantic match: Chloe_Trip (similarity: 0.473)
Found semantic match: Scuba_Hobby (similarity: 0.298)
Found semantic match: Travel_Budget (similarity: 0.251)

🔍 Step 2: Exploring graph from initial matches
  Exploring from: Chloe_Trip
    Related: Chloe_Trip <--[{relationship}]-- Sister_Pet_Sitting
  Exploring from: Sister_Pet_Sitting
    Related: Sister_Pet_Sitting --[affected_by]--> Chloe_Trip
    Related: Sister_Pet_Sitting <--[{relationship}]-- Dog_Arlo
  Exploring from: Dog_Arlo
    Related: Dog_Arlo --[has_behavior]--> Arlo_Anxiety
    Related: Dog_Arlo --[cared_by]--> Sister_Pet_Sitting
  Exploring from: Scuba_Hobby
    Related: Scuba_Hobby --[constrained_by]--> Recent_Ear_Infection
  Exploring from: Travel_Budget
    Related: Travel_Budget --[influences]--> Hotel_Preference
  Exploring from: Arlo_Anxiety
    Related: Arlo_Anxiety <--[{relationship}]-- Dog_Arlo
  Exploring from: Recent_Ear_Infection
    Related: Recent_Ear_Infection --[leads_to]--> Diving_Restriction
    Related: Recent_Ear_Infection <--[{relationship}]-- Scuba_Hobby
  Exploring from: Hotel_Preference
    Related: Hotel_Preference --[relates_to]--> Expensive_Boutique
    Related: Hotel_Preference <--[{relationship}]-- Travel_Budget
  Exploring from: Diving_Restriction
    Related: Diving_Restriction <--[{relationship}]-- Recent_Ear_Infection
  Exploring from: Expensive_Boutique
    Related: Expensive_Boutique <--[{relationship}]-- Hotel_Preference

Step 3: Evaluating 10 candidates for relevance
  LLM selected 4 relevant memories

📋 RESULTS:
1. Scuba_Hobby: I love scuba diving and want to do it on my next beach holiday.
2. Travel_Budget: I have a strict travel budget of $1000 for my next trip.
3. Hotel_Preference: I prefer boutique hotels over large chain hotels.
4. Diving_Restriction: My doctor advised me to avoid diving for at least 6 weeks after an ear infection.

======================================================================
 EXAMPLE 2: I want to go scuba diving on my next vacation.
======================================================================

Processing query: 'I want to go scuba diving on my next vacation.'

Step 1: Performing initial semantic search
Found semantic match: Scuba_Hobby (similarity: 0.842)
Found semantic match: Travel_Budget (similarity: 0.351)
Found semantic match: Diving_Restriction (similarity: 0.350)

🔍 Step 2: Exploring graph from initial matches
  Exploring from: Scuba_Hobby
    Related: Scuba_Hobby --[constrained_by]--> Recent_Ear_Infection
  Exploring from: Recent_Ear_Infection
    Related: Recent_Ear_Infection --[leads_to]--> Diving_Restriction
    Related: Recent_Ear_Infection <--[{relationship}]-- Scuba_Hobby
  Exploring from: Travel_Budget
    Related: Travel_Budget --[influences]--> Hotel_Preference
  Exploring from: Diving_Restriction
    Related: Diving_Restriction <--[{relationship}]-- Recent_Ear_Infection
  Exploring from: Hotel_Preference
    Related: Hotel_Preference --[relates_to]--> Expensive_Boutique
    Related: Hotel_Preference <--[{relationship}]-- Travel_Budget
  Exploring from: Expensive_Boutique
    Related: Expensive_Boutique <--[{relationship}]-- Hotel_Preference

Step 3: Evaluating 6 candidates for relevance
  LLM selected 4 relevant memories

📋 RESULTS:
1. Scuba_Hobby: I love scuba diving and want to do it on my next beach holiday.
2. Travel_Budget: I have a strict travel budget of $1000 for my next trip.
3. Diving_Restriction: My doctor advised me to avoid diving for at least 6 weeks after an ear infection.
4. Recent_Ear_Infection: I recently had a minor ear infection.

======================================================================
 EXAMPLE 3: What should I know about attending my friend's Italian wedding?
======================================================================

Processing query: 'What should I know about attending my friend's Italian wedding?'

Step 1: Performing initial semantic search
Found semantic match: Italian_Wedding (similarity: 0.642)
Found semantic match: Italian_Food (similarity: 0.383)
Found semantic match: Chloe_Trip (similarity: 0.222)

🔍 Step 2: Exploring graph from initial matches
  Exploring from: Italian_Wedding
    Related: Italian_Wedding --[features]--> Italian_Food
    Related: Italian_Wedding <--[{relationship}]-- Lactose_Intolerance
  Exploring from: Lactose_Intolerance
    Related: Lactose_Intolerance --[complicates]--> Italian_Wedding
  Exploring from: Italian_Food
    Related: Italian_Food <--[{relationship}]-- Italian_Wedding
  Exploring from: Chloe_Trip
    Related: Chloe_Trip <--[{relationship}]-- Sister_Pet_Sitting
  Exploring from: Sister_Pet_Sitting
    Related: Sister_Pet_Sitting --[affected_by]--> Chloe_Trip
    Related: Sister_Pet_Sitting <--[{relationship}]-- Dog_Arlo
  Exploring from: Dog_Arlo
    Related: Dog_Arlo --[has_behavior]--> Arlo_Anxiety
    Related: Dog_Arlo --[cared_by]--> Sister_Pet_Sitting
  Exploring from: Arlo_Anxiety
    Related: Arlo_Anxiety <--[{relationship}]-- Dog_Arlo

Step 3: Evaluating 7 candidates for relevance
  LLM selected 3 relevant memories

📋 RESULTS:
1. Italian_Wedding: My friend's wedding is next month, and it's a traditional Italian feast.
2. Italian_Food: Traditional Italian feasts often feature a lot of cheese and cream-based sauces.
3. Lactose_Intolerance: I'm lactose intolerant and avoid dairy products strictly.

======================================================================
 EXAMPLE 4: Is Japan a good destination for me?
======================================================================

Processing query: 'Is Japan a good destination for me?'

Step 1: Performing initial semantic search
Found semantic match: Japan_Interest (similarity: 0.466)
Found semantic match: Japan_Flight_Duration (similarity: 0.417)
Found semantic match: Cherry_Blossom_Season (similarity: 0.305)

🔍 Step 2: Exploring graph from initial matches
  Exploring from: Japan_Interest
    Related: Japan_Interest --[during]--> Cherry_Blossom_Season
    Related: Japan_Interest --[limited_by]--> Flying_Fear
  Exploring from: Japan_Flight_Duration
    Related: Japan_Flight_Duration <--[{relationship}]-- Flying_Fear
  Exploring from: Flying_Fear
    Related: Flying_Fear --[triggered_by]--> Japan_Flight_Duration
    Related: Flying_Fear <--[{relationship}]-- Japan_Interest
  Exploring from: Cherry_Blossom_Season
    Related: Cherry_Blossom_Season <--[{relationship}]-- Japan_Interest

Step 3: Evaluating 4 candidates for relevance
  LLM selected 4 relevant memories

📋 RESULTS:
1. Japan_Interest: I dream of visiting Japan for the cherry blossom season.
2. Japan_Flight_Duration: Flights to Japan from my home are usually over 12 hours long.
3. Flying_Fear: I'm quite scared of flying long distances.
4. Cherry_Blossom_Season: Cherry blossom season in Japan is typically late March to April.

======================================================================
 EXAMPLE 5: Anything I should know about my dog before I travel?
======================================================================

Processing query: 'Anything I should know about my dog before I travel?'

Step 1: Performing initial semantic search
Found semantic match: Arlo_Anxiety (similarity: 0.314)
Found semantic match: Sister_Pet_Sitting (similarity: 0.299)
Found semantic match: Dog_Arlo (similarity: 0.298)

🔍 Step 2: Exploring graph from initial matches
  Exploring from: Arlo_Anxiety
    Related: Arlo_Anxiety <--[{relationship}]-- Dog_Arlo
  Exploring from: Sister_Pet_Sitting
    Related: Sister_Pet_Sitting --[affected_by]--> Chloe_Trip
    Related: Sister_Pet_Sitting <--[{relationship}]-- Dog_Arlo
  Exploring from: Dog_Arlo
    Related: Dog_Arlo --[has_behavior]--> Arlo_Anxiety
    Related: Dog_Arlo --[cared_by]--> Sister_Pet_Sitting
  Exploring from: Chloe_Trip
    Related: Chloe_Trip <--[{relationship}]-- Sister_Pet_Sitting

Step 3: Evaluating 4 candidates for relevance
  LLM selected 3 relevant memories

📋 RESULTS:
1. Arlo_Anxiety: Arlo gets very anxious if left with strangers for too long.
2. Sister_Pet_Sitting: I only trust my sister, Chloe, to watch Arlo when I travel.
3. Chloe_Trip: Chloe is planning a big trip to Europe for all of July.

======================================================================
 EXAMPLE 6: What are the best ways to learn Spanish quickly?
======================================================================

Processing query: 'What are the best ways to learn Spanish quickly?'

Step 1: Performing initial semantic search
Found semantic match: Language_Learning (similarity: 0.667)
Found semantic match: Learning_Spanish (similarity: 0.490)
Found semantic match: Company_Expansion (similarity: 0.200)

🔍 Step 2: Exploring graph from initial matches
  Exploring from: Language_Learning
    Related: Language_Learning <--[{relationship}]-- Learning_Spanish
  Exploring from: Learning_Spanish
    Related: Learning_Spanish --[method]--> Language_Learning
    Related: Learning_Spanish <--[{relationship}]-- Company_Expansion
  Exploring from: Company_Expansion
    Related: Company_Expansion --[motivates]--> Learning_Spanish

Step 3: Evaluating 3 candidates for relevance
  LLM selected 2 relevant memories

📋 RESULTS:
1. Language_Learning: Immersion is the best way to learn a language quickly.
2. Learning_Spanish: I want to learn Spanish to improve my career prospects.

======================================================================
 EXAMPLE 7: What are the best hotels in Tokyo?
======================================================================

Processing query: 'What are the best hotels in Tokyo?'

Step 1: Performing initial semantic search
Found semantic match: Expensive_Boutique (similarity: 0.352)
Found semantic match: Hotel_Preference (similarity: 0.287)
Found semantic match: Japan_Interest (similarity: 0.257)

🔍 Step 2: Exploring graph from initial matches
  Exploring from: Expensive_Boutique
    Related: Expensive_Boutique <--[{relationship}]-- Hotel_Preference
  Exploring from: Hotel_Preference
    Related: Hotel_Preference --[relates_to]--> Expensive_Boutique
    Related: Hotel_Preference <--[{relationship}]-- Travel_Budget
  Exploring from: Japan_Interest
    Related: Japan_Interest --[during]--> Cherry_Blossom_Season
    Related: Japan_Interest --[limited_by]--> Flying_Fear
  Exploring from: Travel_Budget
    Related: Travel_Budget --[influences]--> Hotel_Preference
  Exploring from: Cherry_Blossom_Season
    Related: Cherry_Blossom_Season <--[{relationship}]-- Japan_Interest
  Exploring from: Flying_Fear
    Related: Flying_Fear --[triggered_by]--> Japan_Flight_Duration
    Related: Flying_Fear <--[{relationship}]-- Japan_Interest
  Exploring from: Japan_Flight_Duration
    Related: Japan_Flight_Duration <--[{relationship}]-- Flying_Fear

Step 3: Evaluating 7 candidates for relevance
  LLM selected 3 relevant memories

📋 RESULTS:
1. Travel_Budget: I have a strict travel budget of $1000 for my next trip.
2. Hotel_Preference: I prefer boutique hotels over large chain hotels.
3. Japan_Interest: I dream of visiting Japan for the cherry blossom season.


================================================================================
 HALLUCINATION TESTS (Should find no/minimal relevant memories)
================================================================================

======================================================================
 HALLUCINATION TEST 1: Tell me about my cat Felix.
======================================================================

Processing query: 'Tell me about my cat Felix.'

Step 1: Performing initial semantic search
Found semantic match: Dog_Arlo (similarity: 0.267)
Found semantic match: Sister_Pet_Sitting (similarity: 0.196)
Found semantic match: Arlo_Anxiety (similarity: 0.191)

🔍 Step 2: Exploring graph from initial matches
  Exploring from: Dog_Arlo
    Related: Dog_Arlo --[has_behavior]--> Arlo_Anxiety
    Related: Dog_Arlo --[cared_by]--> Sister_Pet_Sitting
  Exploring from: Sister_Pet_Sitting
    Related: Sister_Pet_Sitting --[affected_by]--> Chloe_Trip
    Related: Sister_Pet_Sitting <--[{relationship}]-- Dog_Arlo
  Exploring from: Arlo_Anxiety
    Related: Arlo_Anxiety <--[{relationship}]-- Dog_Arlo
  Exploring from: Chloe_Trip
    Related: Chloe_Trip <--[{relationship}]-- Sister_Pet_Sitting

Step 3: Evaluating 4 candidates for relevance
  LLM selected 0 relevant memories

📋 RESULTS (Hallucination Test):
✓ Correctly found no relevant memories - system did not hallucinate at the retrieval stage.

======================================================================
 HALLUCINATION TEST 2: What car do I drive?
======================================================================

Processing query: 'What car do I drive?'

Step 1: Performing initial semantic search
Found semantic match: Dog_Arlo (similarity: 0.139)
Found semantic match: Learning_Spanish (similarity: 0.132)
Found semantic match: Sister_Pet_Sitting (similarity: 0.115)

🔍 Step 2: Exploring graph from initial matches
  Exploring from: Dog_Arlo
    Related: Dog_Arlo --[has_behavior]--> Arlo_Anxiety
    Related: Dog_Arlo --[cared_by]--> Sister_Pet_Sitting
  Exploring from: Learning_Spanish
    Related: Learning_Spanish --[method]--> Language_Learning
    Related: Learning_Spanish <--[{relationship}]-- Company_Expansion
  Exploring from: Sister_Pet_Sitting
    Related: Sister_Pet_Sitting --[affected_by]--> Chloe_Trip
    Related: Sister_Pet_Sitting <--[{relationship}]-- Dog_Arlo
  Exploring from: Arlo_Anxiety
    Related: Arlo_Anxiety <--[{relationship}]-- Dog_Arlo
  Exploring from: Language_Learning
    Related: Language_Learning <--[{relationship}]-- Learning_Spanish
  Exploring from: Company_Expansion
    Related: Company_Expansion --[motivates]--> Learning_Spanish
  Exploring from: Chloe_Trip
    Related: Chloe_Trip <--[{relationship}]-- Sister_Pet_Sitting

Step 3: Evaluating 7 candidates for relevance
  LLM selected 0 relevant memories

📋 RESULTS (Hallucination Test):
✓ Correctly found no relevant memories - system did not hallucinate at the retrieval stage.

======================================================================
 HALLUCINATION TEST 3: When is my mother's birthday?
======================================================================

Processing query: 'When is my mother's birthday?'

Step 1: Performing initial semantic search
Found semantic match: Italian_Wedding (similarity: 0.200)
Found semantic match: Chloe_Trip (similarity: 0.145)
Found semantic match: Cherry_Blossom_Season (similarity: 0.118)

🔍 Step 2: Exploring graph from initial matches
  Exploring from: Italian_Wedding
    Related: Italian_Wedding --[features]--> Italian_Food
    Related: Italian_Wedding <--[{relationship}]-- Lactose_Intolerance
  Exploring from: Italian_Food
    Related: Italian_Food <--[{relationship}]-- Italian_Wedding
  Exploring from: Lactose_Intolerance
    Related: Lactose_Intolerance --[complicates]--> Italian_Wedding
  Exploring from: Chloe_Trip
    Related: Chloe_Trip <--[{relationship}]-- Sister_Pet_Sitting
  Exploring from: Cherry_Blossom_Season
    Related: Cherry_Blossom_Season <--[{relationship}]-- Japan_Interest
  Exploring from: Sister_Pet_Sitting
    Related: Sister_Pet_Sitting --[affected_by]--> Chloe_Trip
    Related: Sister_Pet_Sitting <--[{relationship}]-- Dog_Arlo
  Exploring from: Japan_Interest
    Related: Japan_Interest --[during]--> Cherry_Blossom_Season
    Related: Japan_Interest --[limited_by]--> Flying_Fear
  Exploring from: Dog_Arlo
    Related: Dog_Arlo --[has_behavior]--> Arlo_Anxiety
    Related: Dog_Arlo --[cared_by]--> Sister_Pet_Sitting
  Exploring from: Flying_Fear
    Related: Flying_Fear --[triggered_by]--> Japan_Flight_Duration
    Related: Flying_Fear <--[{relationship}]-- Japan_Interest
  Exploring from: Arlo_Anxiety
    Related: Arlo_Anxiety <--[{relationship}]-- Dog_Arlo

Step 3: Evaluating 11 candidates for relevance
  LLM selected 0 relevant memories

📋 RESULTS (Hallucination Test):
✓ Correctly found no relevant memories - system did not hallucinate at the retrieval stage.


================================================================================
 TRUE FACT TESTS (Should find specific relevant memories)
================================================================================

======================================================================
 TRUE FACT TEST 1: Do I have any dietary restrictions?
======================================================================

Processing query: 'Do I have any dietary restrictions?'

Step 1: Performing initial semantic search
Found semantic match: Lactose_Intolerance (similarity: 0.412)
Found semantic match: Italian_Food (similarity: 0.256)
Found semantic match: Italian_Wedding (similarity: 0.225)

🔍 Step 2: Exploring graph from initial matches
  Exploring from: Lactose_Intolerance
    Related: Lactose_Intolerance --[complicates]--> Italian_Wedding
  Exploring from: Italian_Food
    Related: Italian_Food <--[{relationship}]-- Italian_Wedding
  Exploring from: Italian_Wedding
    Related: Italian_Wedding --[features]--> Italian_Food
    Related: Italian_Wedding <--[{relationship}]-- Lactose_Intolerance

Step 3: Evaluating 3 candidates for relevance
  LLM selected 1 relevant memories

📋 RESULTS:
Found 1 memories:
1. Lactose_Intolerance: I'm lactose intolerant and avoid dairy products strictly.
✓ Correctly found all 1/1 expected memories.

======================================================================
 TRUE FACT TEST 2: Tell me about Arlo.
======================================================================

Processing query: 'Tell me about Arlo.'

Step 1: Performing initial semantic search
Found semantic match: Dog_Arlo (similarity: 0.581)
Found semantic match: Arlo_Anxiety (similarity: 0.516)
Found semantic match: Sister_Pet_Sitting (similarity: 0.414)

🔍 Step 2: Exploring graph from initial matches
  Exploring from: Dog_Arlo
    Related: Dog_Arlo --[has_behavior]--> Arlo_Anxiety
    Related: Dog_Arlo --[cared_by]--> Sister_Pet_Sitting
  Exploring from: Arlo_Anxiety
    Related: Arlo_Anxiety <--[{relationship}]-- Dog_Arlo
  Exploring from: Sister_Pet_Sitting
    Related: Sister_Pet_Sitting --[affected_by]--> Chloe_Trip
    Related: Sister_Pet_Sitting <--[{relationship}]-- Dog_Arlo
  Exploring from: Chloe_Trip
    Related: Chloe_Trip <--[{relationship}]-- Sister_Pet_Sitting

Step 3: Evaluating 4 candidates for relevance
  LLM selected 3 relevant memories

📋 RESULTS:
Found 3 memories:
1. Dog_Arlo: My dog Arlo is a golden retriever.
2. Arlo_Anxiety: Arlo gets very anxious if left with strangers for too long.
3. Sister_Pet_Sitting: I only trust my sister, Chloe, to watch Arlo when I travel.
✓ Correctly found all 3/3 expected memories.

======================================================================
 TRUE FACT TEST 3: What language am I learning?
======================================================================

Processing query: 'What language am I learning?'

Step 1: Performing initial semantic search
Found semantic match: Learning_Spanish (similarity: 0.430)
Found semantic match: Language_Learning (similarity: 0.418)
Found semantic match: Company_Expansion (similarity: 0.162)

🔍 Step 2: Exploring graph from initial matches
  Exploring from: Learning_Spanish
    Related: Learning_Spanish --[method]--> Language_Learning
    Related: Learning_Spanish <--[{relationship}]-- Company_Expansion
  Exploring from: Language_Learning
    Related: Language_Learning <--[{relationship}]-- Learning_Spanish
  Exploring from: Company_Expansion
    Related: Company_Expansion --[motivates]--> Learning_Spanish

Step 3: Evaluating 3 candidates for relevance
  LLM selected 1 relevant memories

📋 RESULTS:
Found 1 memories:
1. Learning_Spanish: I want to learn Spanish to improve my career prospects.
✓ Correctly found 1/2 expected memories. (Partially matched)
