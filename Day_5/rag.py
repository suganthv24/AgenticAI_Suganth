from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


document = '''
In a quiet town surrounded by green hills and flowing rivers, there lived a curious young boy named Ethan. He loved reading books more than anything else. Every afternoon after school, he would visit the town library and spend hours exploring different stories. The library was an old building with tall wooden shelves, dusty corners, and thousands of books collected over many decades.
One rainy evening, while searching for a mystery novel, Ethan noticed a small door hidden behind a bookshelf. He had visited the library hundreds of times before, but he had never seen this door. The door was covered with dust and seemed to have been untouched for years. His curiosity grew stronger, and he carefully pushed it open.
Behind the door was a narrow staircase leading underground. Holding a flashlight, Ethan slowly walked down the stairs. At the bottom, he discovered a secret room filled with ancient books, maps, and handwritten manuscripts. The room looked like a forgotten library hidden beneath the town.
As Ethan explored the room, he found a journal written by a librarian who had lived nearly a century earlier. According to the journal, the secret library was created to protect rare books during difficult times. The librarian believed that knowledge was the greatest treasure and wanted to preserve it for future generations.
Over the next few weeks, Ethan visited the secret library whenever he had free time. He carefully read the old manuscripts and learned about the history of his town. He discovered stories about explorers, inventors, and artists who had once lived there. Many of these stories had never been recorded in modern history books.
One day, Ethan found an old map hidden inside a large dictionary. The map pointed to several landmarks around the town and contained mysterious symbols. Excited by the discovery, he decided to investigate. With the help of his best friend Lily, he began following the clues.
Their journey led them through forests, across bridges, and into abandoned buildings. Each location revealed a new clue that brought them closer to solving the mystery. Along the way, they learned valuable lessons about teamwork, patience, and perseverance. Although the challenges were difficult, they never gave up.
Eventually, the clues led them back to the library. Hidden beneath a loose floorboard, they found a small metal box containing letters, photographs, and historical documents. These items revealed important details about the town's founders and explained how the secret library had been established.
The discovery attracted the attention of historians and researchers from nearby cities. Experts visited the town to study the documents and preserve them properly. The local community was amazed to learn about the hidden chapter of their history. The library became famous, and many visitors came to see the place where the remarkable discovery had been made.
Ethan and Lily were praised for their determination and curiosity. However, they believed the true reward was not the recognition they received but the knowledge they had gained. They realized that learning and discovery could change the way people understood the world around them.
Years later, Ethan became a historian and dedicated his life to preserving historical records. He often remembered the rainy evening when he had first discovered the hidden door. That moment taught him that curiosity could lead to extraordinary adventures and that knowledge was one of the most valuable treasures anyone could possess.
The story of the lost library became a legend in the town. Teachers shared it with students, parents told it to their children, and visitors heard it during guided tours. The secret library remained open as a museum and educational center, inspiring future generations to explore, learn, and appreciate the importance of preserving history.
The legacy of the hidden library continued for decades. New collections were added, research projects were conducted, and scholars from different countries visited to study the rare materials. What had once been a forgotten room beneath an old building became a symbol of curiosity, learning, and the power of knowledge.
The town itself also changed because of the discovery. More people became interested in reading, education, and historical research. Schools organized visits to the library, and students participated in projects that encouraged them to explore their local history. As a result, the community developed a stronger appreciation for its cultural heritage.
The lost library reminded everyone that valuable discoveries are often hidden in unexpected places. Sometimes, all it takes is a curious mind, a willingness to explore, and the courage to open a door that others have overlooked. Through Ethan's adventure, people learned that knowledge grows when it is shared and preserved for future generations.
'''
chunks = document.split(".")
chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks)

query = "What happened to the secret library after its discovery?"

query_embedding = model.encode(query)

similarity = cosine_similarity(
    [query_embedding],
    embeddings
)
best_match_index = similarity.argmax()

retreived_chunk = chunks[best_match_index]

prompt = f'''
context: {retreived_chunk}
question: {query}
'''

answer = f'''
Based on the Context,
Answer: {retreived_chunk}
'''