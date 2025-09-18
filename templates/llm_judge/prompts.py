#Raw Correctness
CORRECTNESS = """
You are an expert evaluation system given the following components: an answer and a reference response. 
If the answer is correct relative to the reference response, return a score of 1. 
If the answer is not correct relative to the reference response, return a score of 0. 

Answer: {answer}
Reference Response: {reference_response}
"""

#Context Relevancy Prompt
CONTEXT_RELEVANCY = """
You are an expert evaluation system given the following components: a question and a context. 
If the context is relevant to the question, return a score of 1. 
If the context is not relevant to the question, return a score of 0. 

Question: {user_query}
Context: {retrieved_chunk}
"""

#Main Idea Prompt
MAIN_IDEA = """
You are an expert evaluation system given the following components: a question and a claim. 
If the claim directly answers any part of the question, return a score of 1. 
If the claim does not directly answer any parts of the question, return a score of 0. 

Question: {user_query}
Claim: {claim}
"""

#Claim in Reference Prompt
CLAIM_CONTAINED_REFERENCE = """
You are an expert evaluation system given the following components: a claim and a reference response. 
If the reference answer supports or directly mentions the claim, return a score of 1. 
If the reference answer does not support the claim, return a score of 0. 

Claim: {claim}
Reference Response: {reference_response}
"""

#Claim Contradicts Reference Prompt
CLAIM_CONTRADICTS_REFERENCE = """
You are an expert evaluation system given the following components: a claim and a reference response. 
If the claim directly condradicts the reference answer, return a score of 1. 
If the claim does not directly contradict the reference answer, return a score of 0. 

Claim: {claim}
Reference Response: {reference_response}
"""

#Context Supports Claim Prompt
CONTEXT_SUPPORTED_CLAIM = """
You are an expert evaluation system given the following components: a claim and a context. 
If the information in the claim is supported by the context, return a value of 1. 
If the information in the claim is not suppported by the context, return a value of 0. 

Claim: {claim}
Context Information: {context_info}
"""