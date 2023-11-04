from langchain.document_loaders import UnstructuredCSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
import os
from dotenv import load_dotenv, find_dotenv
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain


def main():
    print("Starting to Load Data")
    load_data()


def load_data():
    try:
        loader = UnstructuredCSVLoader("./RFPQuestion.csv", mode="elements")
        data = loader.load()
        print(f'You have {len(data)} documents in your data')
        print(
            f'There are {len(data[0].page_content)} characters in your document')
        chunk_data(data)
    except Exception as err:
        print(err)


def chunk_data(data):
    print("Chunking...")

    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100, chunk_overlap=0)

        chunks = text_splitter.split_documents(data)
        print(f'Now you have {len(chunks)} documents')

        # do_Embeddings(chunks)
        generate_Natural_Language_Response(chunks)
    except Exception as ex:
        print(ex)


def generate_answer_from_pinecone(chunks):
    # only uses pinecone for an answer not NLP

    print("Doing Embeddings...")

    try:
        load_dotenv(find_dotenv())
        openAI_key = os.environ.get("OPENAI_KEY")
        pinecne_api_key = os.environ.get("PINECONE_API_KEY")
        indexName = os.environ.get("INDEX_NAME")
        embeddings = OpenAIEmbeddings(openai_api_key=openAI_key)

        # initialize pinecone
        pinecone.init(
            api_key=pinecne_api_key,
            environment=os.environ.get("PINECONE_ENVIRONMENT"),
        )
        index_name = indexName

        docsearch = Pinecone.from_texts([t.page_content for t in chunks],
                                        embeddings,
                                        index_name=index_name)

        query = "what is a club"
        results = docsearch.similarity_search(query)
        print(results)
    except Exception as ex:
        print(ex)


def generate_Natural_Language_Response(chunks):
    print("Doing Natural Language Search...")
    try:
        load_dotenv(find_dotenv())
        openAI_key = os.environ.get("OPENAI_KEY")
        pinecne_api_key = os.environ.get("PINECONE_API_KEY")
        indexName = os.environ.get("INDEX_NAME")
        embeddings = OpenAIEmbeddings(openai_api_key=openAI_key)

        llm = OpenAI(temperature=0, openai_api_key=openAI_key)
        chain = load_qa_chain(llm, chain_type="stuff")
        query = "how many clients does a club have"

        # initialize pinecone
        pinecone.init(
            api_key=pinecne_api_key,
            environment=os.environ.get("PINECONE_ENVIRONMENT"),
        )

        index_name = indexName
        docsearch = Pinecone.from_texts([t.page_content for t in chunks],
                                        embeddings,
                                        index_name=index_name)

        docs = docsearch.similarity_search(query)
        answer = chain.run(input_documents=docs, question=query)
        print(answer)
    except Exception as err:
        print(err)


if __name__ == "__main__":
    main()
