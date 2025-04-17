import httpx

async def fetch_pdf(pdf_url: str):
    async with httpx.AsyncClient() as client:
        response = await client.get(pdf_url)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch PDF: {response.status_code}")
        print("Retrieved successfully")
        return response.content 