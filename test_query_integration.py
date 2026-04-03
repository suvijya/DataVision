import asyncio
import pandas as pd
from app.services.session_manager import Session
from app.services.data_analysis import process_query_with_llm
import os
from dotenv import load_dotenv

load_dotenv()

async def test_real_query():
    # Create a dummy dataframe
    df = pd.DataFrame({
        'Sales': [100, 200, 300, 400, 500],
        'Date': pd.date_range(start='2023-01-01', periods=5),
        'Category': ['A', 'B', 'A', 'B', 'A']
    })
    
    # Mock a session
    metadata = {
        'filename': 'test.csv',
        'shape': list(df.shape),
        'columns': list(df.columns)
    }
    session = Session(session_id="test_session", dataframe=df, metadata=metadata)
    
    query = "What is the total sales?"
    print(f"Testing query: '{query}' with model {os.getenv('LLM_MODEL')}")
    
    try:
        response_data, message, exec_time = await process_query_with_llm(session, query)
        print(f"✅ Success! Execution time: {exec_time:.2f}s")
        print(f"Message: {message}")
        print(f"Response Type: {response_data.response_type}")
    except Exception as e:
        print(f"❌ Failed with exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_real_query())
