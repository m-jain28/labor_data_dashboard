import pandas as pd
print ("Pandas:",pd.__version__)
from dotenv import load_dotenv
import os

load_dotenv()
api_key=os.getenv("CENSUS_API_KEY")
print(api_key)