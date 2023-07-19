import openai,sys
# import spacy
import tiktoken # requies python3.9
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
# from sentence_transformers import SentenceTransformer
# from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
idx = pd.IndexSlice

# Set up the OpenAI API client
openai.api_key = 'OPENAI_API_KEY_HERE'


# need something to check if pickle exists
# DOC2VEC = pickle.load(open("gensim_doc2vec_text8.pickle",'rb'))
# DOC2VEC = gensim.downloader.load("glove-wiki-gigaword-50")
# try:
#     DOC2VEC = spacy.load("en_core_web_lg")
# except:
#     print("SpaCy pretrained model en_core_web_lg is missing. Attempting to download for current and future use.")
#     import os
#     os.system("python3.9 -m spacy download en_core_web_lg")
#     DOC2VEC = spacy.load("en_core_web_lg")

def tokenize(s):
    """
    Input a string and return a list of integers representing the string tokens as determined by Tiktoken's 'cl100k_base' encoding.
    :param s: string to be tokenized
    :return: list containing integers representing the tokens in s.
    """
    return tiktoken.get_encoding("cl100k_base").encode(s)

def detokenize(tokens):
    """
    Input a list of integers return a string determined by Tiktoken's 'cl100k_base' encoding.
    :param tokens: list containing integers representing the tokens in s.
    :retrun: string
    """
    return tiktoken.get_encoding("cl100k_base").decode(tokens)

def getTokenCount(s):
    """
    Input a string and return an integer representing the number of tokens in the string according to Tiktoken's 'cl100k_base' encoding.
    :param s: string to be tokenized
    :return: integer representing the number of tokens in s.
    """
    return len(tokenize(s))



# def gensimEmbed(s):
#     """
#     Uses the OpenAI Embedding API to embed a string or list of strings and return Numpy array containing the vector embedding.
#     This function used the 'text-embedding-ada-002' embedding model.
#     :param s: str or list of strings or Pandas.DataFrame
#     :param tokenLimit: int (optional). Default 8191. Represents the API token limit.
#     :return: numpy.array representing the vector embedding of s
#     """
#     def query(s):
#         return DOC2VEC(s).vector
#     if type(s) == str:
#         return query(s)
#     # otherwise, s is a list of strings.
#     if type(s) == list:
#         return np.vstack([query(ss) for ss in s])
#     return gensimEmbed(s["strings"].to_list())

def openAiEmbed(s,tokenLimit=8191):
    """
    Uses the OpenAI Embedding API to embed a string or list of strings and return Numpy array containing the vector embedding.
    This function used the 'text-embedding-ada-002' embedding model.
    :param s: str or list of strings or Pandas.DataFrame
    :param tokenLimit: int (optional). Default 8191. Represents the API token limit.
    :return: numpy.array representing the vector embedding of s
    """
    def query(s):
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=s,
        )
        out = np.array([x["embedding"] for x in response["data"]])
        if len(out) == 1:
            out = out[0]
        return out
    
    if type(s) == str:
        return query(s)
    # otherwise, s is a list of strings.
    if type(s) == list:
        df = pd.DataFrame()
        df["strings"] = s
        s = df
    if not "token count" in s.columns:
        s["token count"] = s["strings"].apply(getTokenCount)
    tokenTotal = s["token count"].sum()
    if tokenTotal < tokenLimit:
        return query(s["strings"].to_list())
    # token limit is exceeded. Need to process in chunks.
    embeddings = []
    currentTokenCount = 0
    chunk = []
    for _,row in s.iterrows():
        s,count = row["strings"],row["token count"]
        if count >= tokenLimit:
            tokens = tokenize(s)
            s = detokenize(tokens[:tokenLimit])
            count = tokenLimit
        if currentTokenCount + count < tokenLimit:
            chunk.append(s)
            currentTokenCount += count
        else:
            embeddings.append(query(chunk))
            chunk = [s]
            currentTokenCount = count
    embeddings.append(query(chunk))
    # print([e.shape for e in embeddings])
    embeddings = np.vstack(embeddings)
    return embeddings

# downloads a bunch of stuff on first use
# def embed_string_using_sbert(s: str or list):
#     """
#     Embeds a string using SBERT after removing stop words.
#     :param s: str or list. A string or list of strings to be embedded
#     :return: Numpy array. Represents the SBERT embedding of s.
#     """
#     model = SentenceTransformer('distilbert-base-nli-mean-tokens')
#     def embed(s):
#         cleaned_s = " ".join([word for word in re.findall(r'\b\w+\b', s) if word.lower() not in ENGLISH_STOP_WORDS])
#         embedding = model.encode([cleaned_s])
#         return np.array(embedding)
#     if type(s) == str:
#         return embed(s)[0]
#     # otherwise, s is a list of strings
#     out = [embed(ss) for ss in s]
#     out = np.vstack(out)
#     return out
    
def embed(*args,**kwargs):
    """
    Wrapper function for the current embedding method.
    """
    # return gensimEmbed(*args,**kwargs)
    return openAiEmbed(*args,**kwargs)
    # return embed_string_using_sbert(*args,**kwargs)


def filterDataFrame(df,prompt,tokenAllowance,promptEmbed=None,):
    """
    Filters a dataframe representing a conversation history or content to the content that is most semantically similar to the prompt string.
    Returns a subset of the dataframe that is within the givne token limit.
    :param df: pandas.DataFrame . The DataFrame representing content to filter. 
         df should include the columns 'token count', and 'embeddeding'.
    :param prompt: str
    :param tokenAllowance: int
    :param promptEmbed: numpy.array (optional). Default None. The word embedding vector representation of the prompt. Embedding is calculated from the OpenAi Embedding API if None.
    :return: Pandas.DataFrame representing the most semantically similar content.
    """
    if promptEmbed is None:
        promptEmbed = embed(prompt)
    if df is None:
        df = self.conversation.copy()
    # print(promptEmbed)        
    df["distance"] = df["embedding"].apply(lambda e: cosine(e,promptEmbed))
    df = df.sort_values(by="distance",ascending=True)
    df["cum token count"] = np.cumsum(df["token count"])
    df = df[df["cum token count"] < tokenAllowance]
    return df

class Conversation:
    """
    Object for communicating with the OpenAI API while storing both prompts and responses.
    """
    def __init__(self,
                 # model="gpt-4-32k",
                 model="gpt-4",
                 # model = "gpt-3.5-turbo",
                 verbose=False):
        """
        Returns an instance of the Conversation class.
        :param model: str (optional). The OpenAI Chat API model to use for the conversation. Default 'gpt-4-32k'.
        :param verbose: boolean (optional). Default False. If True, then print status updates during calls to Conversation methods. Set to True for debugging.
        :return: Conversation instance
        """
        self.setModel(model)
        self.verbose = verbose
        self.conversation = pd.DataFrame(data=[],columns=["role","content","token count",])
        self.tokenCount = 0

    def setModel(self,model):
        """
        Set the OpenAI Chat API model.
        This function sets self.model to the string name of the API model to use and
        sets self.tokenLimit to the token limit corresponding to the model.
        :param self:
        :param model: str representing the OpenApi model name.
        """
        modelTokenLimits = {
            "gpt-4":8000,
            "gpt-4-32k":32000,
            "gpt-3.5-turbo":4000,
            "gpt-3.5-turbo-16k":16384,
        }
        self.tokenLimit = modelTokenLimits[model]
        self.model = model

    def resetConversationHistory(self):
        """
        Resets self.conversation to an empty dataframe.
        """
        self.conversation = pd.DataFrame(data=[],columns=["role","content","token count",])

    def getMessagesToSend(self,prompt):
        """
        This function determines which parts of the conversation history and the current prompt can be sent to the OpenAI Chat API given the API's token limits.
        :param prompt: string representing the prompt for the API.
        :return: integer representing the prompt's token count
        :return: list representing the conversation history to send to the API
        :return: None or Numpy array representing the embedding of the prompt if the conversation history exceeds the API's token limits.
        """
        promptTokenCount = getTokenCount(prompt)
        totalTokenCount = self.conversation["token count"].sum()+promptTokenCount
        if self.tokenLimit > totalTokenCount:
            if self.verbose: 
                print("Token Count: %d is within Token Limit: %d" % (totalTokenCount,self.tokenLimit))
            toSend = [row.to_dict() for _,row in self.conversation[["role","content"]].iterrows()]
            toSend.append({"role":"user","content":prompt})
            return promptTokenCount,toSend,None
        else:
            # need to find the most semantically similar parts of the conversation history.
            tokenAllowance = self.tokenLimit-promptTokenCount
            if self.verbose: 
                print("Token Count: %d is NOT within Token Limit: %d" % (totalTokenCount,self.tokenLimit))
                print("%d tokens remain." % tokenAllowance)
            if not "embedding" in self.conversation.columns:
                if self.verbose:
                    print("Conversation exceed token limit for the first time. Getting embeddings for chat history.")
                E = embed(list(self.conversation["content"]))
                self.conversation["embedding"] = [E[i] for i in range(E.shape[0])]
            # check that embeddings are there. May forget to add embeddings during some debugging efforts.
            I = self.conversation["embedding"].apply(lambda x: type(x) != np.ndarray)
            if np.sum(I) > 0:
                E = embed(list(self.conversation[I]["content"]))
                self.conversation[I]["embedding"] = [E[i] for i in range(E.shape[0])]
            # get most semantically similar parts of the conversation history
            promptEmbed = embed(prompt)

            df = filterDataFrame(self.conversation,prompt,tokenAllowance,promptEmbed,)
            df = df[["role","content"]]
            toSend = [row.to_dict() for _,row in df[["role","content"]].iterrows()]
            toSend.append({"role":"user","content":prompt})
            #self.conversation = self.conversation.append({"role":"user","content":prompt,"token count":promptTokenCount,"embedding":promptEmbed},ignore_index=True)
            return promptTokenCount,toSend,promptEmbed
        

        
    def queryApi(self,prompt):
        """
        Sends prompt to the OpenAI API and then receives response from API using
        openai.ChatCompletion.create
        Stores the prompts and responses in self.messages.
        The model used by the Chat API is determined by self.model.
        If the conversation exceeds the model's token limit, then the conversation history is converted to embeddings using the embed function and the most semantically similar prompts and responses are used in the current prompt until the API token limit is reached.
        

        :param prompt: str. The prompt to be sent to the OpenAI API.
        :return: str. The response from the OpenAI API.
        """
        promptTokenCount,toSend,promptEmbed = self.getMessagesToSend(prompt)
        if promptEmbed is None:
            self.conversation = self.conversation.append({"role":"user","content":prompt,"token count":promptTokenCount},ignore_index=True)
        else:
            self.conversation = self.conversation.append({"role":"user","content":prompt,"token count":promptTokenCount,"embed":promptEmbed},ignore_index=True)            
        
        if self.verbose:
            print("To be sent:")
            print(toSend)
        completion = openai.ChatCompletion.create(
            model=self.model,
            messages=toSend,
        )
        response = completion.choices[0].message.content
        if "embedding" in self.conversation.columns:
            self.conversation = self.conversation.append(
                {
                    "role":"system","content":response,
                    "token count":getTokenCount(response),
                    "embedding":embed(response),
                },
                ignore_index=True
            )
        else:
            self.conversation = self.conversation.append(
                {
                    "role":"system","content":response,
                    "token count":getTokenCount(response),
                },
                ignore_index=True
            )
        return response


def chatInteract(user_input=None):
    C = Conversation()
    while True:
        if user_input is None:
            user_input = input("\nYou: ")
        if user_input.lower().strip() in ["done","stop","finish"]:
            break
        if "[use gpt-4]" in user_input:
            C.setModel("gpt-4-32k")
        elif "[use gpt-3]" in user_input:
            C.setModel("gpt-3.5-turbo")
        user_input = user_input.replace("[use gpt-4]","").replace("[use gpt-3]","")
        reply = C.queryApi(user_input)
        print("\n%s: %s" % (C.model,reply))
        user_input = None
    

if __name__ == "__main__":
    chatInteract()
