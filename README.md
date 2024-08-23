# Simplified focus group chat app using RAG
A proof of concept. Can we use online reviews for a product in place of in-person focus groups? 

Focus groups are a great way to get feedback from customers, but there are a few challeneges:
1. Groups of customers can be small, therefore opinions might not be representative.
2. The environment might suppress honest feelings in some cases.
3. There are usually only a limited number of analysts that can join a session.

Using online reviews with a retrieval-augmented generative model we might be able to replicate the experience. By allowing a user to ask a question about the product, the model can retrieve relevant comments from customers and summarise as an answer.


## Dataset

For this poc we will use amazon reviews downloaded from kaggle using this url: `https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews?resource=download`. As we will be replicating a focus group for a single product, I have selected only the product id with the highest number of reviews which happens to be a packet of cookies: `where 'PorductId' == "B007JFMH8M"`.


## How to run

1. create a virtual env `python3 -m venv .venv`.
2. activate `source .venv/bin/activate`.
3. install rquirements `pip install -r requirements.txt`.
4. install Ollama on your system, you can download from their site: `https://ollama.com/download`. If you're using wsl on windows, make sure you install there using `curl -fsSL https://ollama.com/install.sh | sh`.
5. start an ollama server in a seperate terminal `ollama serve`.
6. pull your desired llm eg. `ollama pull llama2`.
7. run the flask app in a different terminal `python3 app.py`.
8. query the model eg. `curl --request POST --url http://localhost:8080/query --header 'Content-Type: application/json' --data '{ "query": "What do you think about the cookies flavour?" }`

9. Expected response eg. `"message": "Based on the reviews provided, here are some opinions on the flavor of the cookies:\n\n* Some reviewers found the flavor to be lacking, with one stating that it was \"nothing remarkable\" and another saying it was \"too bland.\"\n* A few reviewers mentioned that they would like the cookies to have more flavor, specifically requesting for more spices such as cinnamon, nutmeg, or chocolate.\n* One reviewer found the cookies to be too heavy in consistency, which may affect the overall flavor experience.\n* Some reviewers enjoyed the soft and chewy texture of the cookies, with one stating that it was \"just what they wanted.\"\n* A few reviewers mentioned that the cookies tasted like homemade cookies, which could be seen as a positive aspect of their flavor.\n\nOverall, the opinions on the flavor of the cookies are mixed, with some finding them to be bland and lacking in flavor, while others enjoyed their soft and chewy texture."`

## Next steps

As this is a first iteration, there are other things to try. In terms of prompt engineering it might be interesting to play around with personalities or adding additional demographic data to generate more response types rather than just summarizing reviews by adding to the context.  We could create a synthetic conversation between a marketer and customers if we decided to represent individuals as agents.