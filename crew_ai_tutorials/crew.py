from crewai import Crew
from agents import TravelAgent
from tasks import TravelTasks

class TripCrew:
    def __init__(self, origin, cities, date_range, interests):
        self.origin = origin
        self.cities = cities
        self.date_range = date_range
        self.interests = interests

    def run(self):
        """
            Executes the travel planning process by coordinating the travel agents and tasks.

            The process involves:
            1. Initlilized the specilized travel Agent and tasks.
            2. Assigning the relevent task to dedicated Agents
            3. create a crew to co-ordinate and execute the task
            4. running the crew to generate the structured output for the trip planning.

            Return:
                str: Acoomprehensive travel plan including the selected city and a detailed 7-day itinerary.    

        """
        travel_agent = TravelAgent()
        travel_task = TravelTasks()

        ### Create agent instance
        
        itinerary_agent = travel_agent.expert_travel_agent()
        destination_analysr = travel_agent.city_selection_agent()
        local_tour_guide = travel_agent.local_tour_guide_agent()

        # Assign Task to Agent

        itinerary_task = travel_task.plan_itinary(
            itinerary_agent,
            self.cities,
            self.date_range,
            self.interests
        )

        destination_selection_task = travel_task.identify_city(
            destination_analysr,
            self.origin,
            self.cities,
            self.interests,
            self.date_range
        )

        city_guide_task = travel_task.gether_city_info(
            local_tour_guide,
            self.cities,
            self.date_range,
            self.interests
        )

        # Create and run the Crew

        travel_crew = Crew(
            agents=[itinerary_agent, destination_analysr, local_tour_guide],
            tasks=[itinerary_task, destination_selection_task, city_guide_task],
            verbose=True,
        )

        result = travel_crew.kickoff()
        return str(result)




