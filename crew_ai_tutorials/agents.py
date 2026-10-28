from tabnanny import verbose
from crewai import Agent, LLM
from textwrap import dedent

from tools.calculator_tools import CalculatorTools
from tools.search_tools import SearchTools


class TravelAgent:
    def __init__(self):
        """
            Initializes the TravelAgents class by setting up two OpenAI GPT models:
            - GPT-3.5 model (`gpt-3.5-turbo`) for generating text-based responses.
            - Both models are initialized with a temperature of 0.7 to control randomness
              in responses.

            The models will be used by the different agents (travel planner,
            city selection expert, and local tour guide) for generating responses based
            on their specific goals and tasks.
            """
        self.OpenAIGPT35 = LLM(model="gpt-3.5-turbo", temperature=0.7)
        self.OpenAIGPT4 = LLM(model="gpt-4o-mini", temperature=0.7)
    
    def expert_travel_agent(self):
        """
        Sets up the 'Expert Travel Agent' to help plan a 7-day trip.

        This agent can:
        - Build a detailed 7-day itinerary for the trip.
        - Suggest a budget for the entire journey.
        - Offer packing tips.
        - Share important safety advice.

        The agent uses the following tools:
        - `SearchTools.search_internet`: To look up useful travel info online.
        - `CalculatorTools.calculate`: To handle any necessary calculations (like costs).

        The agent works with the GPT-3.5 model (`self.OpenAIGPT35`) to create responses and complete tasks.

        Returns:
            Agent: The fully configured Expert Travel Agent.
        """
        return Agent(
            role="Expert Travel Agent",
            backstory=dedent("""You are an expert travel agent with extensive knowledge of global destinations,
                                travel logistics, and cultural insights. Your role is to assist travelers in planning
                                comprehensive trips, including creating detailed itineraries, budgeting, and providing
                                safety advice. You are skilled at using online resources to gather the latest information
                                about destinations, accommodations, and activities."""),
            goal=f"""Plan a 7-day trip with day-by-day details, including a budget, packing tips, and safety recommendations.""",
            llm=self.OpenAIGPT4,
            tools=[SearchTools.search_internet, CalculatorTools.calculate],
            verbose=True
        )

    def city_selection_agent(self):
        """
        Creates a 'City Selection agent' agent to help choose the best travel destinations.

        This agent can:
        - Analyze travel data to recommend cities based on various factors.
        - Take into account weather, season, prices, and the preferences of the traveler.

        The agent uses the following tool:
        - `SearchTools.search_internet`: To gather city-related information online.

        The agent uses the GPT-3.5 model (`self.OpenAIGPT35`) to generate recommendations and respond to queries.

        Returns:
            Agent: The fully set-up City Selection Expert.
        """
        return Agent(
            role="City Selection Expert",
            backstory=dedent("""You are a city selection expert with deep knowledge of global cities, their climates,
                                cultural events, and travel costs. Your role is to assist travelers in choosing the best
                                city for their trips based on their preferences and the time of year. You are adept at
                                analyzing data and using online resources to provide accurate and up-to-date information."""),
            goal=f"""Analyze multiple potential destinations by evaluating weather conditions, seasonal events, and travel costs.
                    The selected city should align with the traveler's interests and provide the best overall experience.""",
            llm=self.OpenAIGPT4,
            tools=[SearchTools.search_internet],
            verbose=True
        )

    def local_tour_guide_agent(self):
        """
        Sets up a 'Local Tour Guide' agent to provide detailed information about specific cities.

        This agent can:
        - Offer insights into local attractions, cultural events, and travel logistics.
        - Provide recommendations based on the traveler's interests and the city's offerings.

        The agent uses the following tool:
        - `SearchTools.search_internet`: To find relevant information about the city online.

        The agent uses the GPT-4 model (`self.OpenAIGPT4`) to generate detailed city guides and respond to queries.

        Returns:
            Agent: The fully configured Local Tour Guide.
        """
        
        return Agent(
            role="Local Tour Guide",
            backstory=dedent("""You are a local tour guide with extensive knowledge of your city, including its attractions,
                                cultural events, and travel logistics. Your role is to provide travelers with comprehensive
                                information about the city, helping them make informed decisions about what to see and do.
                                You are skilled at using online resources to gather the latest information about local events
                                and attractions."""),
            goal=f"""Provide detailed information about specific cities, including key attractions, cultural events,
                    and travel logistics. Tailor recommendations based on the traveler's interests.""",
            llm=self.OpenAIGPT4,
            tools=[SearchTools.search_internet],
            verbose=True
        )

