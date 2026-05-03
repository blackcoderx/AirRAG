from google import genai


class GeminiGenerator:
    """Wraps Gemini model for grounded answer generation (RAG synthesis).

    Takes retrieved context chunks and generates a factual answer based ONLY on that context.
    This prevents hallucination from the model's training data.

    """

    def __init__(self, api_key: str, model: str):
        """Initialize Gemini client for text generation.

        Model: gemini-2.0-flash (fast multimodal generation).
        """
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def generate(self, query: str, context_chunks: list[str]) -> str:
        """Generate grounded answer from retrieved context chunks.

        Args:
            query: User's question
            context_chunks: Text chunks retrieved from Qdrant (ranked by similarity)
        """
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
        response = self._client.models.generate_content(
            model=self._model, contents=[prompt]
        )

        return "" if response.text is None else response.text
