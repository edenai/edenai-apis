from edenai_apis.utils.exception import ProviderException
from linkup import LinkupClient
from typing import Optional, List


class LinkupSearch:
    
    
    def __init__(self):
        self.client = LinkupClient()
    
    def text__search(
        self,
        query: str,
        depth: str="deep",
        model: Optional[str] = None,
        texts: Optional[List[str]] = None,
        similarity_metric: str= "cosine"
    ):
        try:
            payload = {
                "query": query,
                "depth": depth,
                "output_type": "searchResults",
            }
            return self.client.search(**payload)
        except Exception as e:
            print("DEBUG - Exception occurred:", str(e))
            raise ProviderException(f"Error during Linkup API call: {str(e)}")
