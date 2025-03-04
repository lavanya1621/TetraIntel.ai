import os
import time
import streamlit as st
from typing import Annotated, List, Dict, Any
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import json


load_dotenv()


class ResearchItem(BaseModel):
    query: str = Field(description="The search query used")
    source: str = Field(description="Source of the information")
    content: str = Field(description="Content retrieved from the source")

class State(TypedDict):
    messages: Annotated[list, add_messages]
    research_results: List[ResearchItem]
    research_summary: str  
    follow_up_questions: List[str]  
    current_depth: int 


@st.cache_resource
def get_llm():
    return ChatGroq(
        model="llama3-70b-8192",  
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2
    )

@st.cache_resource
def get_tavily_tool():
    return TavilySearchResults(
        max_results=4, 
        tavily_api_key=os.getenv("TAVILY_API_KEY")
    )

# Define the four agents for our system
def research_coordinator(state: State) -> Dict:
    """Plans research and coordinates the information gathering process."""
    messages = state["messages"]
    user_query = messages[-1].content if messages[-1].type == "human" else "Need research"
    
    system_prompt = """You are a Research Coordinator. Your role is to:
    1. Analyze the user's query to understand their information needs
    2. Break it down into 2-3 specific, targeted search queries that will yield comprehensive information
    3. Focus on questions that will provide diverse perspectives on the topic
    
    Format your response as a JSON list of search queries.
    """
    
    prompt = f"{system_prompt}\n\nUser Query: {user_query}\n\nGenerate search queries:"
    
    llm = get_llm()
    response = llm.invoke(prompt)
    
    try:
        search_queries = json.loads(response.content)
        if not isinstance(search_queries, list):
            search_queries = [user_query]  
    except:
      
        search_queries = [user_query]
    
    # Provide status update to user
    progress_message = f"Planning research approach with {len(search_queries)} targeted queries..."
    st.info(progress_message)
    
    return {
        "follow_up_questions": search_queries,
        "current_depth": 0,
        "messages": state["messages"] + [AIMessage(content=progress_message)]
    }

def information_gatherer(state: State) -> Dict:
    """Gathers information using Tavily Search based on queries."""
    follow_up_questions = state.get("follow_up_questions", [])
    current_depth = state.get("current_depth", 0)
    max_depth = 2  
    if not follow_up_questions or current_depth >= max_depth:
        
        return {"research_results": state.get("research_results", [])}
    
 
    current_query = follow_up_questions[0]
    remaining_questions = follow_up_questions[1:]
    

    tavily_tool = get_tavily_tool()
    search_results = tavily_tool.invoke(current_query)
    
 
    processed_results = []
    for result in search_results:
        processed_results.append(
            ResearchItem(
                query=current_query,
                source=result.get("url", "Unknown source"),
                content=result.get("content", "No content available")
            )
        )

    existing_results = state.get("research_results", [])
    updated_results = existing_results + processed_results
    
   
    progress_message = f"Researching: '{current_query}' - Found {len(processed_results)} relevant sources..."
    st.info(progress_message)
    
    return {
        "research_results": updated_results,
        "follow_up_questions": remaining_questions,
        "current_depth": current_depth + 1,
        "messages": state["messages"] + [AIMessage(content=progress_message)]
    }

def knowledge_synthesizer(state: State) -> Dict:
    """Synthesizes research findings into a cohesive summary."""
    results = state.get("research_results", [])
    
    if not results:
        return {"research_summary": "No research results found."}
    
   
    research_text = ""
    for i, result in enumerate(results):
        research_text += f"Source {i+1} [{result.source}]:\n{result.content}\n\n"
    
    system_prompt = """You are a Knowledge Synthesizer. Your task is to:
    1. Analyze the research data from multiple sources
    2. Identify key themes, facts, and insights
    3. Resolve contradictions and note different perspectives
    4. Create a cohesive, comprehensive summary that captures the essence of the research
    5. Maintain factual accuracy and cite sources where appropriate
    
    Focus on creating a valuable synthesis that will serve as the foundation for answering the user's query.
    """
    
    prompt = f"{system_prompt}\n\nResearch Data:\n{research_text}\n\nSynthesize this information:"
    
    llm = get_llm()
    response = llm.invoke(prompt)
    
   
    progress_message = f"Synthesizing information from {len(results)} sources..."
    st.info(progress_message)
    
    return {
        "research_summary": response.content,
        "research_results": results,  
        "messages": state["messages"] + [AIMessage(content=progress_message)]
    }

def answer_composer(state: State) -> Dict:
    """Composes a comprehensive final answer using the research summary."""
    messages = state["messages"]
    original_messages = [m for m in messages if m.type != "ai" or not any(x in m.content for x in ["Planning research", "Researching:", "Synthesizing"])]
    user_query = original_messages[-1].content if original_messages[-1].type == "human" else "Need answer"
    
    research_summary = state.get("research_summary", "No research summary available.")
    research_results = state.get("research_results", [])
    
    system_prompt = """You are an Answer Composer for a Deep Research AI System. Your role is to:
    1. Craft a comprehensive, well-structured response to the user's original query
    2. Base your answer on the synthesized research provided
    3. Present information in a clear, logical flow
    4. Include relevant facts, nuanced perspectives, and key insights
    5. Acknowledge any limitations in the available information
    6. Format your response for clarity and readability (use markdown formatting with headings, bullet points where appropriate)
    7. At the end of your response, include a 'Sources' section that lists all URLs used in the research
    
    Your answer should be thorough yet concise, and directly address what the user wanted to know.
    """
    
    
    sources_list = "\n\nSources:\n"
    unique_sources = set()
    for item in research_results:
        unique_sources.add(item.source)
    
    for i, source in enumerate(unique_sources):
        sources_list += f"{i+1}. {source}\n"
    
    prompt = f"{system_prompt}\n\nOriginal User Query: {user_query}\n\nResearch Summary:\n{research_summary}\n\n{sources_list}\n\nCompose a comprehensive answer:"
    
    llm = get_llm()
    response = llm.invoke(prompt)
    
    return {"messages": original_messages + [AIMessage(content=response.content)]}

def should_continue_research(state: State) -> str:
    """Determines if more research is needed or if we should move to synthesis."""
    follow_up_questions = state.get("follow_up_questions", [])
    current_depth = state.get("current_depth", 0)
    max_depth = 2
    
    if not follow_up_questions or current_depth >= max_depth:
        return "synthesize"
    else:
        return "continue"

def create_research_graph():
    graph_builder = StateGraph(State)

   
    graph_builder.add_node("research_coordinator", research_coordinator)
    graph_builder.add_node("information_gatherer", information_gatherer)
    graph_builder.add_node("knowledge_synthesizer", knowledge_synthesizer)
    graph_builder.add_node("answer_composer", answer_composer)

  
    graph_builder.add_edge(START, "research_coordinator")
    graph_builder.add_edge("research_coordinator", "information_gatherer")
    graph_builder.add_conditional_edges(
        "information_gatherer",
        should_continue_research,
        {
            "continue": "information_gatherer",
            "synthesize": "knowledge_synthesizer"
        }
    )
    graph_builder.add_edge("knowledge_synthesizer", "answer_composer")
    graph_builder.add_edge("answer_composer", END)

  
    return graph_builder.compile()


def main():
    st.set_page_config(
        page_title="Deep Research AI",
        page_icon="🔍",
        layout="wide",
    )

   
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
    }
    .research-section {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
    </style>
    """, unsafe_allow_html=True)


    st.markdown("<h1 class='main-header'>🔍 Welcome to TetraIntel.ai </h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Four Agents, One Mission - Deeper Research, Sharper Insights</p>", unsafe_allow_html=True)

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'research_in_progress' not in st.session_state:
        st.session_state.research_in_progress = False

 
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            st.chat_message("assistant").markdown(message["content"])

    if not st.session_state.research_in_progress:
        user_input = st.chat_input("Ask a research question...")
        
        if user_input:
            
            st.chat_message("user").write(user_input)
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            st.session_state.research_in_progress = True
            
            progress_container = st.container()
            
            with progress_container:
                st.markdown("<div class='research-section'>", unsafe_allow_html=True)
                st.subheader("Research in Progress...")
                progress_bar = st.progress(0)
                
            
                try:
                    graph = create_research_graph()
                    
                    initial_state = {
                        "messages": [HumanMessage(content=user_input)],
                        "research_results": [],
                        "research_summary": "",
                        "follow_up_questions": [],
                        "current_depth": 0
                    }
                    
                    with st.spinner("Researching your query..."):
                       
                        progress_bar.progress(25)
                        time.sleep(0.5)
                        
                      
                        final_state = graph.invoke(initial_state)
                 
                        progress_bar.progress(100)
                     
                        final_answer = final_state["messages"][-1].content
                    
                        st.chat_message("assistant").markdown(final_answer)
                        st.session_state.chat_history.append({"role": "assistant", "content": final_answer})
                        
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.session_state.chat_history.append({"role": "assistant", "content": f"I encountered an issue while processing your request. Please try again or rephrase your question.\n\nError: {str(e)}"})
                
                st.markdown("</div>", unsafe_allow_html=True)
            
                st.session_state.research_in_progress = False
                st.rerun()

if __name__ == "__main__":
    main()