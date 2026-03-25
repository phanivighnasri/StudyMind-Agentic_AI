from config import SIMILARITY_THRESHOLD

def check_syllabus(query, syllabus_db):
    results = syllabus_db.similarity_search_with_score(query, k=1)

    if not results:
        return False

    doc, score = results[0]

    if score > SIMILARITY_THRESHOLD:
        return False

    return True
