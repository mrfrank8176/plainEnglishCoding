from chatGPT import *
import numpy as np
import pandas as pd
from IPython import get_ipython
from IPython.core.magic import Magics, magics_class, line_magic, cell_magic
ipython = get_ipython()


def list_object_names():
    """
    List all variable names in memory.
    :return: list of strings
    """
    names = [
            name for name in ipython.user_ns.keys() \
            if type(name) == str \
            and name[0] != '_' \
            and not name in ["In","Out","get_ipython","exit","quit","ipython",]
        ]
    return names

def list_module_names():
    """
    List all variable names in memory that are modules.
    :return: list of strings.
    """
    ipython = get_ipython()
    return [name for name in list_object_names() if type(ipython.user_ns[name]) == type(pd)]

def list_class_names():
    """
    List all variable names in memory that are class definitions.
    :return: list of strings.
    """
    ipython = get_ipython()
    return [name for name in list_object_names() if isinstance(ipython.user_ns[name],type)]

def list_objects_in_memory(names=None):
    """
    List all objects in memory along with their data type. If the object is a function,
    show its docstring. If the object is a Pandas DataFrame, show the data types of each column and examples from the column.
    :param names: list or None (optional). Default None. If a list, the list contains strings of 
        variable names to be included in the description of Python's current memory. 
        This can significantly reduce the size of the prompt sent to the OpenAI API
        if many objects exist in memory.
    :return: list of strings representing the objects in memory and their descriptions.
    """
    ipython = get_ipython()
    if names is None:
        names = list_object_names()
    objectDescriptions = []
    for name in names:
        obj = ipython.user_ns[name]
        if callable(obj) and not isinstance(obj, type):  # exclude classes
            doc = obj.__doc__
            if not doc is None:
                doc = doc.strip()
            objectDescriptions.append({"name":name,"type":"function","description":f"{name}: (Function)"+f"  Docstring: {doc}\n"})
        elif type(obj) == type(pd):
            # if module
            if name == obj.__name__:
                objectDescriptions.append({"name":name,"type":"module","description":"%s\n" % name})
            else:
                objectDescriptions.append({"name":name,"type":"module","description":"%s imported as %s\n" % (obj.__name__,name)})
        elif isinstance(obj, pd.DataFrame):
            s = f"{name}: (Pandas DataFrame)\n"
            s += "\tColumn data types:\n"
            for column, dtype in obj.dtypes.items():
                s += f"\t\t{column}: {dtype}"
            if type(obj.index) in [pd.core.indexes.multi.MultiIndex,pd.core.indexes.base.Index]:
                s += "\n\tIndex level data types:\n"
                if len(obj.index.names) == 1:
                    s += f"\t\t{obj.index.names[0]}: {obj.index.dtype}"
                else:
                    for level,dtype in obj.index.dtypes.items():
                        s += f"\t\t{level}: {dtype}"
            s += "\n\t>>>%s.head(5)\n" % name
            s += ''.join(map(lambda s: "\t%s\n" % s, str(obj.head(5)).split("\n")))
            s += "\n"
            objectDescriptions.append({"name":name,"type":"dataframe","description":s})
        elif isinstance(obj,type):
            # object is a class definition
            s = "%s: (custom class)\n" % name
            s += "\tClass Docstring: %s" % obj.__doc__
            s += "\tClass Methods:\n"
            for k,v in vars(obj).items():
                if not k in ["__module__","__doc__","__dict__ ","__weakref__"]:
                    s += "\t\t%s  Docstring: %s\n" % (k,v.__doc__)
            objectDescriptions.append({"name":name,"type":"class","description":s})
        elif isinstance(obj, (int, float, complex)):
            s = f"{name}: ({type(obj).__name__})"
            s += " Value: %s\n" % str(obj)
            objectDescriptions.append({"name":name,"type":"numeric","description":s})
        else:
            # maybe an instance of a custom class.... maybe something generic like a list or dict
            try:
                s = "%s: (%s)\n" % (name,type(obj))
                s += "\tParameters:\n"
                for k,v in vars(obj).items():
                    s += "\t\t%s: %s\n" % (k,v)
            except Exception as e:
                s = f"{name}: ({type(obj).__name__})\n"
            objectDescriptions.append({"name":name,"type":str(type(obj)),"description":s})
    objectDescriptions = pd.DataFrame(objectDescriptions).set_index(["name","type"])
    return objectDescriptions

class gptCoder(Conversation):
    """
    Class for using the OpenAI Chat API to answer coding questions.
    This class inherits from the Conversation class.
    Additional parameters: self.lastResponse (str) representing the last coding response from the API. Or None if no response has been received yet.
    """
    def __init__(self,*args,model="gpt-4",**kwargs):
        """
        Initialize the class.
        Passes *args and **kwargs to Conversation.__init__
        """
        self.lastResponse = ''
        self.lastCodingQuestion = ''
        super().__init__(*args,model=model,**kwargs)
        # self.setModel("gpt-3.5-turbo")

    def getPromptHeader(self):
        """
        Creates a prompt string to be sent to the OpenAI API.
        The prompt specifies that there will be a coding question in Python and details
        rules for the GPT's response.
    
        :return: str with the prompt for the OpenAI API.
        """
        # " - Never refer to yourself as 'AI', you are a coding assistant.\n"+\
        # " - Be polite and respectful in your response.\n"+\
        # " - If you are not sure about something, don't guess.\n"+\
        promptHeader = "You are a coding assistant. You are helping the user complete the code they are trying to write "+\
            "while following these requirements:\n"+\
            " - Only complete the code in the FOCAL CELL.\n"+\
            " - Make sure Python modules are not already imported in ACTIVE MEMORY before importing the module in your code.\n"+\
            " - Only put the completed code in a function if the user explicitly asks you to, otherwise just complete the code in the FOCAL CELL.\n"+\
            " - Provide a docstring describing inputs and outputs for any function you define in your code.\n"+\
            " - Provide code that is intelligent, correct, efficient, and readable.\n"+\
            " - Keep your responses short and to the point.\n"+\
            " - Provide your code and completions. Never format your code as markdown code blocks.\n"+\
            " - Never ask the user for a follow up. Do not include pleasantries at the end of your response.\n" +\
            " - Never summarise the new code you wrote at the end of your response.\n"+\
            " - Do not wrap your code in a coding block. Your response should only contain code.\n"+\
            "I will list each object in Python's ACTIVE MEMORY using '>>' and provide a description for each object:\n"
            # self.list_objects_in_memory(**kwargs)+\
            # "Focal cell:\n"+\
            # user_input
        return promptHeader

    def getContext(self,prompt,header=None,variables=None,):
        if header is None:
            header = self.getPromptHeader()
        headerTokenCount = getTokenCount(header)
        promptTokenCount = getTokenCount(prompt)
        if not variables is None:
            variables = np.union1d(variables,list_module_names())
        memory = list_objects_in_memory(names=variables)
        memory["description"] = memory["description"].apply(lambda s: " >> %s" % s)
        memory["token count"] = memory["description"].apply(getTokenCount)
        memory["role"] = "user"
        if self.tokenLimit > headerTokenCount+memory["token count"].sum()+promptTokenCount:
            if self.verbose:
                print("Token limit not exceeded. Proceeding with API call.")
            context = pd.concat(
                [
                    pd.DataFrame([{"role":"user","content":header,"token count":headerTokenCount}]),
                    memory.reset_index().rename(columns={"description":"content"})[["role","content","token count"]],
                ]
            )
        else:
            if self.verbose:
                print(
                    "Prompt will exceed token limit for API call. "+\
                    "Need to filter to most semantically similar objects (+modules & classes) in memory"
                )
            memory = memory.reset_index()
            # print(memory)
            # print("+=++++++")
            #modulesInMemory = memory.loc[idx[:],["module","class"],:]
            #otherObjects = memory.drop(index=["module","class"],level="type")
            modulesInMemory = memory[memory["type"].isin(["module","class"])].copy()
            modulesInMemory["description"] = modulesInMemory.apply(
                lambda row: row["description"] if row["token count"] <= 500 else row["description"].split(":")[0],
                axis=1
            )
            otherObjects = memory[~memory["type"].isin(["module","class"])].copy()
            tokenAllowance = self.tokenLimit - (headerTokenCount+modulesInMemory["token count"].sum()+promptTokenCount)
            # print("remaining token allowance: %d" % tokenAllowance)
            # print(otherObjects.head())
            embeddings = embed(otherObjects.rename(columns={"description":"strings"}))
            # embeddings = embed(otherObjects["description"].to_list())
            otherObjects["embedding"] = [embeddings[i] for i in range(embeddings.shape[0])]
            # otherObjects["embedding"] = otherObjects["description"].apply(embed)
            otherObjects = filterDataFrame(otherObjects,prompt,tokenAllowance,promptEmbed=None,)
            context = pd.concat(
                [
                    pd.DataFrame([{"role":"user","content":header,"token count":headerTokenCount}]),
                    modulesInMemory.reset_index().rename(columns={"description":"content"})[["role","content","token count"]],
                    otherObjects.reset_index().rename(columns={"description":"content"})[["role","content","token count"]],
                    # pd.DataFrame([{"role":"user","content":codingQuestion,"token count":codingQuestionTokenCount}])
                ]
            )
        return context

    def filterToCode(self,s):
        """
        Looks for a ChatGPT style coding block in a string and filters the string to just the code.
        :param s: string
        :return: str
        """
        if self.verbose:
            print("Before code filtering:")
            print(s)
        for block in ['```',"'''",'"""']:
            if "%spython" % block in s:
                s = s.split(block)[1]
        if s[:6] == "python":
            # remember newline character
            s = s[7:]
        if self.verbose:
            print("After code filtering:")
            print(s)
        return s
    
    def getFullPrompt(self,s,variables=None,header=None):
        """
        Combines input string with header string to produce a string that should be within token limits for the API call.
        :param s: str
        :param variables: list or None (optional). Default None. If list, then it should contain strings representing the names
                                 of variables in active memory to be included in the API call.
        :param header: str or None (optional). Default None. 
        :return: str representing the string to be sent to the API
        """
        if header is None:
            header = self.getPromptHeader()
        context = self.getContext(s,header=header,variables=variables)
        prompt = ''.join(context["content"].to_list()+[s,])
        self.resetConversationHistory()
        return prompt
        
    def getCode(self,codingQuestion,variables=None,suppress=False):
        """
        Sends coding question to OpenAI Chat API while communicating Python objects in memory.
        Is input 'names' is a list, then the list contains the variables to be sent to the API. 
        Otherwise, all objects are described.
        :param codingQuestion: str (required) coding question
        :param variables: list or None (optional). Default None. 
        :param suppress: boolean (optional). Default False. Suppressing printing the response.
        :return: str. Response from OpenAI API
        """
        self.lastCodingQuestion = codingQuestion
        codingQuestion = "FOCAL CELL:\n"+codingQuestion
        prompt = self.getFullPrompt(codingQuestion,variables)
        if self.verbose:
            print("Prompt to be sent to OpenAI Chat API:")
            print(prompt)
        response = self.queryApi(prompt)
        response = self.filterToCode(response)
        response = response.strip()
        self.lastResponse = response
        if not suppress:
            print(response)
        return response

    def explain(self,code=None,variables=None,suppress=False):
        """
        Sends a code snippet to the OpenAI Chat API asking for an explanation of the code.
        Information about the objects in Python memory are also provided.
        If code is not provided, then self.lastResponse is used.
        :param code: str (optional). 
        :param variables: None or list of strings (optional). Default None. If list of strings, then strings represent variable names to be included in memory description.
        :param suppress: boolean (optional). Default False. Suppressing printing the response.
        """
        if code is None:
            code = self.lastResponse
        prompt = "Explain the following code:\n%s" % code
        header = "Here is what is in Python's ACTIVE MEMORY:\n"
        prompt = self.getFullPrompt(prompt,header=header)
        response = self.queryApi(prompt)
        response = response.strip()
        if not suppress:
            print(response)
        return response

    def fix(self,error,codingQuestion=None,code=None,variables=None,suppress=False,maxAttempts=0):
        """
        Sends a Python error and code snippet to the OpenAI Chat API asking corrected code.
        Information about the objects in Python memory are also provided.
        If code is not provided, then self.lastResponse is used.
        If maxAttempts > 0, then the function will run the code to check for errors.
        If code produces new errors, then another query will be sent to the API to try to fixe the code.
        :param error: str representing the error message associated with running the code. 
        :param codingQuestion: str or None (optional). The prompt that the code is supposed to answer. Uses self.lastCodingQuestion if None
        :param code: str or None (optional). 
        :param variables: None or list of strings (optional). Default None. If list of strings, then strings represent variable names to be included in memory description.
        :param suppress: boolean (optional). Default False. Suppressing printing the response.
        :param maxAttempts: int (optional). Default 0. Max number of attempts to fix code through calls to the OpenAI Chat API.
        :return: str representing the corrected code.
        """
        if code is None:
            code = self.lastResponse
        if codingQuestion is None:
            codingQuestion = self.lastCodingQuestion
        if len(codingQuestion) == 0:
            prompt = ""
        else:
            prompt = "The code in FOCAL CELL is meant to answer the following prompt: %s\n" % codingQuestion
        prompt += "The code in FOCAL CELL produces the following error: %s\n" % error +\
                "FOCAL CELL:\n%s" % code
        # " - Only put the completed code in a function if the user explicitly asks you to, otherwise just complete the code in the FOCAL CELL.\n"+\
        # " - If you are not sure about something, don't guess.\n"+\
        header = "You are a coding assistant. You are helping the user complete the code they are trying to write "+\
            "while following these requirements:\n"+\
            " - Adapt the code in the FOCAL CELL.\n"+\
            " - Your answer should directly replace the code in FOCAL CELL.\n"+\
            " - Make sure Python modules are not already imported in ACTIVE MEMORY before importing the module in your response.\n"+\
            " - Provide a docstring for any function you define in your response.\n"+\
            " - Provide code that is intelligent, correct, efficient, and readable.\n"+\
            " - Keep your responses short and to the point.\n"+\
            " - Provide your code and completions. Never format your code as markdown code blocks.\n"+\
            " - Never ask the user for a follow up. Do not include pleasantries at the end of your response.\n" +\
            " - Never summarize the code in your response.\n"+\
            " - Never explain your response.\n"+\
            " - Never summerize the code in FOCAL CELL.\n"+\
            " - Do not wrap your code in a coding block. Your response should only contain code.\n"+\
            " - Your response should incorporate any new code into the old code.\n"+\
            "Here is what is in Python's ACTIVE MEMORY:\n"
        context = self.getContext(prompt,header=header,variables=variables)
        self.resetConversationHistory()
        prompt = ''.join(context["content"].to_list())+prompt
        response = self.queryApi(prompt)
        response = self.filterToCode(response)
        response = response.strip()
        for attempt in range(maxAttempts):
            if self.verbose:
                print("Checking that response code runs without errors. Attempt %d (out of %d)" % (attempt,maxAttempts))
            try:
                exec(response)
                if self.verbose:
                    print("Code runs without error.")
                break
            except Exception as e:
                err = "%s: %s" % (str(e.__class__).split("'")[1],e)
                if self.verbose:
                    print("Response produced errors.")
                    print("Code:")
                    print(response)
                    print("\nError: %s" % err)
                self.resetConversationHistory()
                prompt = "The code in FOCAL CELL is meant to answer the following prompt: %s\n" % codingQuestion+\
                "The code in FOCAL CELL produces the following error: %s\n" % err +\
                "FOCAL CELL:\n%s" % response
                prompt = ''.join(context["content"].to_list())+prompt
                response = self.queryApi(prompt)
                response = self.filterToCode(response)
        if not suppress:
            print(response)
        return response

@magics_class
class MyMagics(Magics):

    @cell_magic
    def gpt(self,line,cell):
        self.code(line,cell)
    
    @cell_magic
    def code(self,line,cell):
        #"Magic %%gpt"
        # If using the magic command fails, then try the same prompt using gptCoder directly to debug.
        # Also, sometimes, the text in the cell won't update. This is a Javascript issue and not a Python issue.
        # Fix by (1) saving the notebook and (2) refreshing the browser.
        global GPTCODER
        if not "GPTCODER" in list_object_names():
            GPTCODER = gptCoder()
        runCode = "-r" in line
        line2 = line.replace("-r","")
        if len(line2) == 0:
            names = None
        else:
            names = line2.strip().split()
        response = GPTCODER.getCode(cell,variables=names,suppress=True,)
        # self.shell.set_next_input('# %lmagic\n{}'.format(raw_code), replace=True)
        self.shell.set_next_input(
            '"""\n%%%%code %s\n%s"""\n%s' % (line,"" if cell is None else cell,response),
            replace=True,
        )
        if runCode:
            # if you want to run the code instead.
            self.shell.run_cell(response, store_history=False)

    @cell_magic
    def explain(self,line,cell):
        global GPTCODER
        if not "GPTCODER" in list_object_names():
            GPTCODER = gptCoder()
        if cell is None or len(cell) == 0:
            cell = None
        GPTCODER.explain(code=cell,suppress=False)

    @cell_magic
    def fix(self,line,cell):
        #"Magic %%fix"
        # If using the magic command fails, then try the same prompt using gptCoder directly to debug.
        # Also, sometimes, the text in the cell won't update. This is a Javascript issue and not a Python issue.
        # Fix by (1) saving the notebook and (2) refreshing the browser. 
        global GPTCODER
        if not "GPTCODER" in list_object_names():
            GPTCODER = gptCoder()
        runCode = "-r" in line
        line2 = line.replace("-r","").strip()
        useLastQuestion = "-q" in line2
        line2 = line2.replace("-q","").strip()
        if useLastQuestion:
            codingQuestion = GPTCODER.lastCodingQuestion
        else:
            codingQuestion = ''
        response = GPTCODER.fix(line2,codingQuestion=codingQuestion,code=cell,suppress=True,)
        # self.shell.set_next_input('# %lmagic\n{}'.format(raw_code), replace=True)
        self.shell.set_next_input(
            '"""\n%%%%fix %s\n%s"""\n%s' % (line,"" if cell is None else cell,response),
            replace=True,
        )
        if runCode:
            # if you want to run the code instead.
            self.shell.run_cell(response, store_history=False)

    @cell_magic
    def chat(self,line,cell):
        # global CHATBOT
        # if "CHATBOT" not in list_object_names():
        #     CHATBOT = Conversation()
        # if "-reset" in line:
        #     CHATBOT.resetConversationHistory()
        # response = CHATBOT.queryApi(cell)
        # chatHistory = ''.join(["%s: %s\n----\n" % (row["role"],row["content"]) for _,row in CHATBOT.conversation.iterrows()])
        # print(chatHistory)
        chatInteract(cell)
        

try:
    ip = get_ipython()
    ip.register_magics(MyMagics)
    del ip
    global GPTCODER
    GPTCODER = gptCoder()
except Exception as e:
    pass
