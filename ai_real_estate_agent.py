from typing import Dict, List
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.google import Gemini
from firecrawl import FirecrawlApp
import streamlit as st
from urllib.parse import quote
import os

open_api_key = os.environ.get('OPENAI_API_KEY')
firecrawl_api_key = os.environ.get('FIRECRAWL_API_KEY')

class PropertyData(BaseModel):
    """Schema for property data extraction"""
    building_name: str = Field(description="Name of the building/property", alias="Building_name")
    property_type: str = Field(description="Type of property (commercial, residential, etc)", alias="Property_type")
    location_address: str = Field(description="Complete address of the property")
    price: str = Field(description="Price of the property", alias="Price")
    description: str = Field(description="Detailed description of the property", alias="Description")

class PropertiesResponse(BaseModel):
    """Schema for multiple properties response"""
    properties: List[PropertyData] = Field(description="List of property details")

class LocationData(BaseModel):
    """Schema for location price trends"""
    location: str
    price_per_sqft: float
    percent_increase: float
    rental_yield: float

class LocationsResponse(BaseModel):
    """Schema for multiple locations response"""
    locations: List[LocationData] = Field(description="List of location data points")

class FirecrawlResponse(BaseModel):
    """Schema for Firecrawl API response"""
    success: bool
    data: Dict
    status: str
    expiresAt: str

class PropertyFindingAgent:
    """Agent responsible for finding properties and providing recommendations"""
    
    def __init__(self, firecrawl_api_key: str, openai_api_key: str, model_id: str = "gpt-3.5-turbo"):
        self.agent = Agent(
            model=Gemini(id=model_id, api_key=openai_api_key),
            markdown=True,
            description="I am a real estate expert who helps find and analyze properties based on user preferences."
        )
        self.firecrawl = FirecrawlApp(api_key=firecrawl_api_key)

    def find_properties(
        self, 
        city: str,
        max_price: float,
        property_category: str = "Residential",
        property_type: str = "HDB",
        min_size: int = None,
        max_size: int = None,
        bedrooms: int = None
    ) -> str:
        """Find and analyze properties based on user preferences"""
        formatted_location = city.lower()
        # if hdb - > H , condo -> N , landed -> L
        # if property_category = residential -> res , commercial -> com
        property_map = {
            "Residential": {
                "HDB": "H",
                "Condo": "N",
                "Landed": "L"
            },
            "Commercial": "com"
        }

        pro_type = property_map[property_category]
        if isinstance(pro_type, dict):
            pro_type = pro_type.get(property_type, "H")

        if min_size is None:
            min_size_param = ''
        else:
            min_size_param = min_size
        if max_size is None:
            max_size_param = ''
        else:
            max_size_param = max_size
        if bedrooms is None:
            bedrooms_param = ''
        else:
            bedrooms_param = bedrooms
        urls = [
            f"https://www.propertyguru.com.sg/property-for-sale?listingType=sale&page=1&isCommercial=false&maxPrice={max_price}&freetext={quote(formatted_location)}&bedrooms={bedrooms_param}&minSize={min_size_param}&maxSize={max_size_param}&propertyTypeGroup={pro_type}"
          #  f" https://www.propertyguru.com.sg/property-for-sale?listingType=sale&page=1&isCommercial=true&_freetextDisplay={formatted_location}&priceMax={max_price * 10000000}",
           # f"https://www.99.co/singapore/sale",
           # f"https://housing.com/in/buy/{formatted_location}/{formatted_location}",
            # f"https://www.nobroker.in/property/sale/{city}/{formatted_location}",
        ]
        print(f"URLS: {urls}")
        property_type_prompt = {
            "HDB": "HDB Flats",
            "Condo": "Condominiums",
            "Landed": "Landed Houses"
        }.get(property_type, "HDB Flats")
        
        prompt = f"""Extract ONLY 10 OR LESS different {property_category} {property_type_prompt} from {city} that cost less than {max_price} SGD.
        
        Requirements:
        - Property Category: {property_category} properties only
        - Property Type: {property_type_prompt} only
        - Location: {city}
        - Maximum Price: {max_price} SGD
        - Include complete property details with exact location
        - IMPORTANT: Return data for at least 3 different properties. MAXIMUM 10.
        - Format as a list of properties with their respective details
        - IMPORTANT: include the hyperlink to the each property in the resulted data
        - IMPORTANT: extract the image link of each property
        """
        
        if min_size:
            prompt += f"\n- Minimum Size: {min_size} sq ft"
        if max_size:
            prompt += f"\n- Maximum Size: {max_size} sq ft"
        if bedrooms:
            prompt += f"\n- Number of Bedrooms: {bedrooms}"
        
        raw_response = self.firecrawl.extract(
            urls=urls,
            params={
                'prompt': prompt,
                'schema': PropertiesResponse.model_json_schema()
            }
        )
        
        print("Raw Property Response:", raw_response)
        
        if isinstance(raw_response, dict) and raw_response.get('success'):
            properties = raw_response['data'].get('properties', [])
        else:
            properties = []
            
        print("Processed Properties:", properties)

        
        if len(properties) == 0:
            return "No properties found. Please try again with different search criteria."
        else:
            analysis = self.agent.run(
                f"""As a real estate expert, analyze these properties and market trends:

                Properties Found in json format:
                {properties}

                **IMPORTANT INSTRUCTIONS:**
                1. ONLY analyze properties from the above JSON data that match the user's requirements:
                - Property Category: {property_category}
                - Property Type: {property_type}
                - Maximum Price: {max_price} SGD
                2. DO NOT create new categories or property types
                3. From the matching properties, select 5-6 properties with prices closest to {max_price} SGD

                Please provide your analysis in this format:
                
                🏠 SELECTED PROPERTIES
                • List with number only 5-6 best matching properties with prices closest to {max_price} SGD
                • For each property include:
                - Name and Location
                - Price (with value analysis)
                - Key Features
                - Pros and Cons
                - hyperlink to the property listing

                💰 BEST VALUE ANALYSIS
                • Compare the selected properties based on:
                - Price per sq ft
                - Location advantage
                - Amenities offered

                📍 LOCATION INSIGHTS
                • Specific advantages of the areas where selected properties are located

                💡 RECOMMENDATIONS
                • Top 3 properties from the selection with reasoning
                • Investment potential
                • Points to consider before purchase

                🤝 NEGOTIATION TIPS
                • Property-specific negotiation strategies

                Format your response in a clear, structured way using the above sections.
                """
            )
        
        return analysis.content
    

    def get_location_trends(self, city: str) -> str:
        """Get price trends for different localities in the city"""
        raw_response = self.firecrawl.extract([
            f"https://www.99acres.com/property-rates-and-price-trends-in-{city.lower()}-prffid/*"
        ], {
            'prompt': """Extract price trends data for ALL major localities in the city. 
            IMPORTANT: 
            - Return data for at least 5-10 different localities
            - Include both premium and affordable areas
            - Do not skip any locality mentioned in the source
            - Format as a list of locations with their respective data
            """,
            'schema': LocationsResponse.model_json_schema(),
        })
        
        if isinstance(raw_response, dict) and raw_response.get('success'):
            locations = raw_response['data'].get('locations', [])
    
            analysis = self.agent.run(
                f"""As a real estate expert, analyze these location price trends for {city}:

                {locations}

                Please provide:
                1. A bullet-point summary of the price trends for each location
                2. Identify the top 3 locations with:
                   - Highest price appreciation
                   - Best rental yields
                   - Best value for money
                3. Investment recommendations:
                   - Best locations for long-term investment
                   - Best locations for rental income
                   - Areas showing emerging potential
                4. Specific advice for investors based on these trends

                Format the response as follows:
                
                📊 LOCATION TRENDS SUMMARY
                • [Bullet points for each location]

                🏆 TOP PERFORMING AREAS
                • [Bullet points for best areas]

                💡 INVESTMENT INSIGHTS
                • [Bullet points with investment advice]

                🎯 RECOMMENDATIONS
                • [Bullet points with specific recommendations]
                """
            )
            
            return analysis.content
            
        return "No price trends data available"


def main():
    st.set_page_config(
        page_title="Real Estate Agent for Singapore 🇸🇬 property market",
        page_icon="🏠",
        layout="wide"
    )

    # Try to get keys from environment first
    firecrawl_key = os.environ.get('FIRECRAWL_API_KEY')
    openai_key =  os.environ.get('GOOGLE_API_KEY') #'AIzaSyBFJikVub5H_v7zZdQ-SRCjw1dRcIAYHmw' # os.environ.get('GEMINI_API_KEY')
    st.session_state.model_id = 'gemini-2.0-pro-exp-02-05'

    if not firecrawl_key or not openai_key:
        with st.sidebar:
            st.title("🔑 API Configuration")
            
            st.subheader("🤖 Model Selection")
            model_id = st.selectbox(
                "Choose OpenAI Model",
                options=["gpt-3.5-turbo", "gpt-4o"],
                help="Select the AI model to use. Choose gpt-4o if your api doesn't have access to o3-mini"
            )
            st.session_state.model_id = model_id
        
            


            
            st.divider()
            st.subheader("🔐 API Keys")
            
            # Only show inputs if keys are not in environment
            if not firecrawl_key:
                firecrawl_key = st.text_input(
                "Firecrawl API Key",
                type="password",
                help="Enter your Firecrawl API key"
                )
            
            if not openai_key:
                openai_key = st.text_input(
                "OpenAI API Key", 
                type="password",
                help="Enter your OpenAI API key"
                )
        
    if firecrawl_key and openai_key:
        st.session_state.firecrawl_key = firecrawl_key
        st.session_state.openai_key = openai_key
        st.session_state.property_agent = PropertyFindingAgent(
            firecrawl_api_key=firecrawl_key,
            openai_api_key=openai_key,
            model_id=st.session_state.model_id
        )

    st.title("🏠AI Real Estate Agent for Singapore 🇸🇬 property market")
    st.info(
        """
        Stop wasting time browsing web. Use smart AI agent to find your matching property instance!
        Quick and easy way to find properties in Singapore 🇸🇬 based on your preferences. 
        and location insights.
        """
    )

    col1, col2 = st.columns(2)
    
    with col1:
        city = st.text_input(
            "City",
            placeholder="Enter city name (e.g., Punggol, Downtown Core)",
            help="Enter the city or district where you want to search for properties"
        )
        
        property_category = st.selectbox(
            "Property Category",
            options=["Residential", "Commercial"],
            help="Select the type of property you're interested in"
        )

    with col2:
        max_price = st.number_input(
            "Maximum Price (in SGD)",
            min_value=1000,
            max_value=100000000,
            value=500000,
            step=10000,
            help="Enter your maximum budget in SGD"
        )
        
        property_type = st.selectbox(
            "Property Type",
            options=["HDB", "Condo", "Landed"],
            help="Select the specific type of property"
        )
        
        min_size = st.number_input(
            "Minimum Size (in sq ft)",
            min_value=0,
            max_value=20000,
            placeholder=500,
            value=None,
            step=50,
            help="Enter the minimum size of the property in square feet"
        )
        
        max_size = st.number_input(
            "Maximum Size (in sq ft)",
            min_value=0,
            max_value=30000,
            placeholder=1000,
            value=None,
            step=50,
            help="Enter the maximum size of the property in square feet"
        )
        
        bedrooms = st.number_input(
            "Number of Bedrooms",
            min_value=0,
            max_value=10,
            placeholder=2,
            value=None,
            step=1,
            help="Enter the number of bedrooms"
        )

    if st.button("🔍 Start Search", use_container_width=True):
        if 'property_agent' not in st.session_state:
            st.error("⚠️ Please enter your API keys in the sidebar first!")
            return
            
        if not city:
            st.error("⚠️ Please enter a city name!")
            return
            
        try:
            with st.spinner("🔍 Searching for properties..."):
                property_results = st.session_state.property_agent.find_properties(
                    city=city,
                    max_price=max_price,
                    property_category=property_category,
                    property_type=property_type,
                    min_size=min_size,
                    max_size=max_size,
                    bedrooms=bedrooms
                )
                
                st.success("✅ Property search completed!")
                
                st.subheader("🏘️ Property Recommendations")
                st.markdown(property_results)
                
                st.divider()
                
                # with st.spinner("📊 Analyzing location trends..."):
                #     location_trends = st.session_state.property_agent.get_location_trends(city)
                    
                #     st.success("✅ Location analysis completed!")
                    
                #     with st.expander("📈 Location Trends Analysis of the city"):
                #         st.markdown(location_trends)
                
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")

if __name__ == "__main__":
    main()