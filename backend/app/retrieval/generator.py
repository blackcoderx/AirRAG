from google import genai


class GeminiGenerator:
    def __init__(self, api_key: str, model: str):
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def generate(self, query: str, context_chunks: list[str]) -> str:
        if not context_chunks:
            return "No relevant information found in the knowledge base."
        # Separators make chunk boundaries explicit so the model doesn't blend adjacent chunks
        context = "\n\n---\n\n".join(context_chunks)
        # Grounding instruction prevents the model from answering from training knowledge
        prompt = (
            f"Answer the following question based only on the provided context. "
            f"If the answer is not in the context, say so.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\nAnswer:"
        )
        response = self._client.models.generate_content(model=self._model, contents=[prompt])
        return response.text
