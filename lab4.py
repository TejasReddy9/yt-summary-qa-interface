# --------------------------------------------------------------------
# FIASS ( HNSW ) : AI-Powered YouTube Summarizer, QA Tool with RAG, LangChain, FAISS
# --------------------------------------------------------------------
# Import necessary libraries for the YouTube bot
import gradio as gr
import re  #For extracting video id 
from youtube_transcript_api import YouTubeTranscriptApi  # For extracting transcripts from YouTube videos
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For splitting text into manageable segments
from langchain_community.vectorstores import FAISS  # For efficient vector storage and similarity search
from langchain.chains import LLMChain  # For creating chains of operations with LLMs
from langchain.prompts import PromptTemplate  # For defining prompt templates
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import PromptTemplate


def get_video_id(url):    
    # Regex pattern to match YouTube video URLs
    pattern = r'https:\/\/www\.youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else None

# url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
# video_id = get_video_id(url)
# print(video_id)  # Output: dQw4w9WgXcQ


def get_transcript(url):
    video_id = get_video_id(url)
    ytt_api = YouTubeTranscriptApi()
    transcripts = ytt_api.list(video_id)
    transcript = ""

    for t in transcripts:

        if t.language_code == 'en':
            if t.is_generated:
                # If no transcript has been set yet, use the auto-generated one
                if len(transcript) == 0:
                    transcript = t.fetch()
            else:
                # If a manually created transcript is found, use it (overrides auto-generated)
                transcript = t.fetch()
                break  # Prioritize the manually created transcript, exit the loop
    
    return transcript if transcript else None


# url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
# transcript = get_transcript(url)
# print(transcript)

def process(transcript):
    txt = ""
    
    for i in transcript:
        try:
            txt += f"Text: {i.text} Start: {i.start}\n"
        except KeyError:
            pass
            
    return txt

# url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
# transcript = get_transcript(url)
# formatted_transcript = process(transcript)
# print(formatted_transcript)

def chunk_transcript(processed_transcript, chunk_size=200, chunk_overlap=20):
    # Initialize the RecursiveCharacterTextSplitter with specified chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Split the transcript into chunks
    chunks = text_splitter.split_text(processed_transcript)
    return chunks

# url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
# transcript = get_transcript(url)
# formatted_transcript = process(transcript)
# chunks = chunk_transcript(formatted_transcript)
# print(chunks)


def initialize_llm(model_id):
    return OllamaLLM(model=model_id)

def setup_embedding_model(embed_model):
    return OllamaEmbeddings(model=embed_model)

def create_faiss_index(chunks, embedding_model):
    # """
    # Create a FAISS index from text chunks using the specified embedding model.
    
    # :param chunks: List of text chunks
    # :param embedding_model: The embedding model to use
    # :return: FAISS index
    # """
    # Use the FAISS library to create an index from the provided text chunks
    return FAISS.from_texts(chunks, embedding_model)

def perform_similarity_search(faiss_index, query, k=3):
    # """
    # Search for specific queries within the embedded transcript using the FAISS index.
    
    # :param faiss_index: The FAISS index containing embedded text chunks
    # :param query: The text input for the similarity search
    # :param k: The number of similar results to return (default is 3)
    # :return: List of similar results
    # """
    # Perform the similarity search using the FAISS index
    results = faiss_index.similarity_search(query, k=k)
    return results

def create_summary_prompt():
    # """
    # Create a PromptTemplate for summarizing a YouTube video transcript.
    
    # :return: PromptTemplate object
    # """
    # Define the template for the summary prompt
    template = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an AI assistant tasked with summarizing YouTube video transcripts. Provide concise, informative summaries that capture the main points of the video content.

    Instructions:
    1. Summarize the transcript in a single concise paragraph.
    2. Ignore any timestamps in your summary.
    3. Focus on the spoken content (Text) of the video.

    Note: In the transcript, "Text" refers to the spoken words in the video, and "start" indicates the timestamp when that part begins in the video.<|eot_id|><|start_header_id|>user<|end_header_id|>
    Please summarize the following YouTube video transcript:

    {transcript}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    
    # Create the PromptTemplate object with the defined template
    prompt = PromptTemplate(
        input_variables=["transcript"],
        template=template
    )
    return prompt

def create_summary_chain(llm, prompt, verbose=True):
    # """
    # Create an LLMChain for generating summaries.
    
    # :param llm: Language model instance
    # :param prompt: PromptTemplate instance
    # :param verbose: Boolean to enable verbose output (default: True)
    # :return: LLMChain instance
    # """
    return LLMChain(llm=llm, prompt=prompt, verbose=verbose)

def retrieve(query, faiss_index, k=7):
    # """
    # Retrieve relevant context from the FAISS index based on the user's query.

    # Parameters:
    #     query (str): The user's query string.
    #     faiss_index (FAISS): The FAISS index containing the embedded documents.
    #     k (int, optional): The number of most relevant documents to retrieve (default is 3).

    # Returns:
    #     list: A list of the k most relevant documents (or document chunks).
    # """
    relevant_context = faiss_index.similarity_search(query, k=k)
    return relevant_context

def create_qa_prompt_template():
    # """
    # Create a PromptTemplate for question answering based on video content.

    # Returns:
    #     PromptTemplate: A PromptTemplate object configured for Q&A tasks.
    # """
    
    # Define the template string
    qa_template = """
    You are an expert assistant providing detailed answers based on the following video content.

    Relevant Video Context: {context}

    Based on the above context, please answer the following question:
    Question: {question}
    """
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=qa_template
    )
    return prompt_template


# qa_prompt_template = create_qa_prompt_template()
# context = "This video explains the fundamentals of quantum physics."
# question = "What are the key principles discussed in the video?"
# generated_prompt = qa_prompt_template.format(context=context, question=question)
# print(generated_prompt)

def create_qa_chain(llm, prompt_template, verbose=True):
    # """
    # Create an LLMChain for question answering.

    # Args:
    #     llm: Language model instance
    #         The language model to use in the chain (e.g., WatsonxGranite).
    #     prompt_template: PromptTemplate
    #         The prompt template to use for structuring inputs to the language model.
    #     verbose: bool, optional (default=True)
    #         Whether to enable verbose output for the chain.

    # Returns:
    #     LLMChain: An instantiated LLMChain ready for question answering.
    # """
    return LLMChain(llm=llm, prompt=prompt_template, verbose=verbose)

def generate_answer(question, faiss_index, qa_chain, k=7):
    # """
    # Retrieve relevant context and generate an answer based on user input.

    # Args:
    #     question: str
    #         The user's question.
    #     faiss_index: FAISS
    #         The FAISS index containing the embedded documents.
    #     qa_chain: LLMChain
    #         The question-answering chain (LLMChain) to use for generating answers.
    #     k: int, optional (default=3)
    #         The number of relevant documents to retrieve.

    # Returns:
    #     str: The generated answer to the user's question.
    # """
    relevant_context = retrieve(question, faiss_index, k=k)
    answer = qa_chain.predict(context=relevant_context, question=question)
    return answer





# Initialize an empty string to store the processed transcript after fetching and preprocessing
processed_transcript = ""

def summarize_video(video_url):
    # """
    # Title: Summarize Video

    # Description:
    # This function generates a summary of the video using the preprocessed transcript.
    # If the transcript hasn't been fetched yet, it fetches it first.

    # Args:
    #     video_url (str): The URL of the YouTube video from which the transcript is to be fetched.

    # Returns:
    #     str: The generated summary of the video or a message indicating that no transcript is available.
    # """
    global fetched_transcript, processed_transcript
    
    
    if video_url:
        # Fetch and preprocess transcript
        fetched_transcript = get_transcript(video_url)
        processed_transcript = process(fetched_transcript)
    else:
        return "Please provide a valid YouTube URL."

    if processed_transcript:
        # step 1: set model
        model_id = 'mistral:7b'

        # step 2: init llm
        llm = initialize_llm(model_id)

        # Step 3: Create the summary prompt and chain
        summary_prompt = create_summary_prompt()
        summary_chain = create_summary_chain(llm, summary_prompt)

        # Step 4: Generate the video summary
        summary = summary_chain.run({"transcript": processed_transcript})
        return summary
    else:
        return "No transcript available. Please fetch the transcript first."
    

def answer_question(video_url, user_question):
    # """
    # Title: Answer User's Question

    # Description:
    # This function retrieves relevant context from the FAISS index based on the user’s query 
    # and generates an answer using the preprocessed transcript.
    # If the transcript hasn't been fetched yet, it fetches it first.

    # Args:
    #     video_url (str): The URL of the YouTube video from which the transcript is to be fetched.
    #     user_question (str): The question posed by the user regarding the video.

    # Returns:
    #     str: The answer to the user's question or a message indicating that the transcript 
    #          has not been fetched.
    # """
    global fetched_transcript, processed_transcript

    # Check if the transcript needs to be fetched
    if not processed_transcript:
        if video_url:
            # Fetch and preprocess transcript
            fetched_transcript = get_transcript(video_url)
            processed_transcript = process(fetched_transcript)
        else:
            return "Please provide a valid YouTube URL."

    if processed_transcript and user_question:
        # Step 1: Chunk the transcript (only for Q&A)
        chunks = chunk_transcript(processed_transcript)

        # step 2: set model
        model_id = 'mistral:7b'

        # step 3: init llm
        llm = initialize_llm(model_id)

        # Step 4: Create FAISS index for transcript chunks (only needed for Q&A)
        embedding_model = setup_embedding_model('nomic-embed-text:latest')
        faiss_index = create_faiss_index(chunks, embedding_model)

        # Step 5: Set up the Q&A prompt and chain
        qa_prompt = create_qa_prompt_template()
        qa_chain = create_qa_chain(llm, qa_prompt)

        # Step 6: Generate the answer using FAISS index
        answer = generate_answer(user_question, faiss_index, qa_chain)
        return answer
    else:
        return "Please provide a valid question and ensure the transcript has been fetched."
    
with gr.Blocks() as interface:
    # Input field for YouTube URL
    video_url = gr.Textbox(label="YouTube Video URL", placeholder="Enter the YouTube Video URL")
    
    # Outputs for summary and answer
    summary_output = gr.Textbox(label="Video Summary", lines=5)
    question_input = gr.Textbox(label="Ask a Question About the Video", placeholder="Ask your question")
    answer_output = gr.Textbox(label="Answer to Your Question", lines=5)

    # Buttons for selecting functionalities after fetching transcript
    summarize_btn = gr.Button("Summarize Video")
    question_btn = gr.Button("Ask a Question")

    # Display status message for transcript fetch
    transcript_status = gr.Textbox(label="Transcript Status", interactive=False)

    # Set up button actions
    summarize_btn.click(summarize_video, inputs=video_url, outputs=summary_output)
    question_btn.click(answer_question, inputs=[video_url, question_input], outputs=answer_output)

# Launch the app with specified server name and port
interface.launch(server_name="127.0.0.1", server_port=7860)