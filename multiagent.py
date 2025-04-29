
from phi.agent import Agent
from phi.model.groq import Groq
from dotenv import load_dotenv
from phi.tools.serpapi_tools import SerpApiTools
from phi.tools.crawl4ai_tools import Crawl4aiTools
import streamlit as st
import os

load_dotenv()

serpapi_tools = SerpApiTools(api_key=os.getenv("SERPAPI_API_KEY"))


def content_writing(topic):

    content_planner_agent = Agent(
           name = "Content Planner",
           role="Content Planning",
           model = Groq(id="deepseek-r1-distill-llama-70b"),
           tools = [serpapi_tools, Crawl4aiTools()],
           show_tool_calls = True,
           markdown=True,
           instructions=["You are an expert Content Planner. Your task is to plan an engaging and factually accurate content on the given topic. Always use the tools to search for something on web. You collect information that helps the audience learn something and make informed decisions. Your work is the basis for the Content Writer's work to write an article on this topic. Prioritize the latest trends and noteworthy news on this topic. Develop a detailed content outline including an introduction, key points etc. Include SEO keywords and relevant data or sources."],
           debug_mode = True
           )
    
    
    content_writer_agent = Agent (
           name = "Content Writer",
           role = "Content Writing",
           model = Groq(id="deepseek-r1-distill-llama-70b"),
           tools = [serpapi_tools, Crawl4aiTools()],
           show_tool_calls = True,
           markdown=True,
           instructions=["You are an expert Content Writer. Your task is to write an insightful and factually accurate article about the topic. Your work is based on the work of the Content Planner who provides an outline and relevant context about the topic. You follow the main objectives and direction of the outline as provided by the Content Planner. Always use the tools to search for something on web. You also provide objective and impartial insights and back them up with information provided by the Content Planner. Use the content plan to craft a compelling blog post on this topic. Incorporate SEO keywords. Sections/Subtitles are properly named in an engaging manner. Ensure the post is structured with an engaging introduction, insightful body and a summarizing conclusion. The well-written blog post should be in markdown format, ready for publication."],
           debug_mode = True
           )
    
    
    editor_agent = Agent (
           name = "Editor",
           role = "Editing the content",
           model = Groq(id="deepseek-r1-distill-llama-70b"),
           tools = [serpapi_tools, Crawl4aiTools()],
           show_tool_calls = True,
           markdown=True,
           instructions=["You are an expert Content Editor. Your task is to edit the blog post written by Content Writer. Your goal is to review the blog post to ensure that it follows journalistic best practices, provides balanced viewpoints when providing opinions or assertions and also avoids major controversial opinions. Please check the given blog post for grammatical errors. It should be a well-written blog post in markdown format, ready for publication."],
           debug_mode = True
           )
    
    
    
    agent_team = Agent(
        model = Groq(id="deepseek-r1-distill-llama-70b"),
        team = [content_planner_agent, content_writer_agent, editor_agent],
        instructions=["Do content planning for this topic: " + topic, "do content writing for the given topic based on the content planning", "edit the written content for the topic and make it look very organized and professional"],
        show_tool_calls = True,
        markdown=True,  
        )
    
    
    output = agent_team.run("Do web search using tools and write a blog/article on the given topic. Please always include sources of information.", stream=False)
    

    return output.content



def main():
    
    html_temp = """
    <div style="background-color:brown;padding:8px">
    <h2 style="color:white;text-align:center;">AI App with Multi-Agent Workflows for Content Writing</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)    

    st.image("logo4.jpg", width=500)
   
    topic = st.text_input("**Topic**","")

    
    st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #DD3300;
        color:#eeffee;
    }
    </style>""", unsafe_allow_html=True)

    if st.button("Write Content"):
        
        results = content_writing(topic)
        print(results)
        
        st.success('Results {}'.format(results))
    

if __name__=='__main__':
    main()

