import json
import os

from openai import OpenAI

ANALYSIS_SYSTEM_PROMPT = '''
You will be provided with a Context (delimited with [[CONTEXT]] and [[END CONTEXT]]), that can contain Pages (delimited with <<<START PAGE : N >>> and <<<END PAGE : N>>>) and Tables (delimited with <<START TABLE : N>> and <<END TABLE : N>>).

You will also be provided with a question (delimited with [[QUESTION]]), that will ask you to extract specific information from the Context.

Your task is to extract the information requested in the question from the Context. You will always return the extracted answer along with the page number where the answer was found as well as the table number if the answer is within a table in a json format.

Everything in the "answer" json response should be a string and not a structured output.

{
    'answer': The extracted answer (string),
    'score' : The score of the answer (int),
    'document' : The document title (string),
    'page': The page number where the answer was found (int),
    'table': The table number where the answer was found (int)
}

If an answer is wholly contained within a table, then print out the entire table.

Finally, you will output a "score" in the json that rates (out of 4) whether the information required to answer the question is present in the context. The score should be calculated as follows:
1 - Does not disclose anything related to the guidance
2 - Has disclosed some information related to the guidance
3 - Answers most of the guidance but not all
4 - Clearly answers every part of the guidance

If the score is 1, then do not return any text. Otherwise, return the extracted answer, the score, the page number, and the table number (if applicable) in the json format.

Below are 5 examples. Your output should be what follows the [[ANSWER]] tag in the examples.

EXAMPLE 1:

[[CONTEXT]]

<<<DOCUMENT: document1, START PAGE : 44>>>

Diversity metrics
Following, diversity indicators are presented, broken down into both, Adtran Networks SE group and the entire 
Adtran Holdings, Inc. group, and into total employees vs. employees in management, gender and age, respectively. 
During 2024, we will continue our actions on diversity and inclusion, which will include more detailed target 
definitions. All data presented here is as of December 31, 2023. Data for Adtran Holdings, Inc. group is highlighted 
in light gray, it was not subject to the limited-assurance engagement.
Adtran Networks SE group 2023 Adtran Holdings, Inc. group 2023
Headcount % Headcount %
Males total 1,680 77.6 2,587 76.4
Males in management 338 15.6 493 14.6
Females total 483 22.3 790 23.3
Females in management 55 2.5 94 2.8
Not declared total 3 0.1 8 0.2
Not declared in management 0 0 1 0.03
Management Adtran Networks SE group 2023 Adtran Holdings, Inc. group 2023
Males not in management 1,342 2,094
Females not in management 428 696
Not declared not in management 3 7
Total not in management 1,773 2,797
Males in management 338 493
Females in management 55 94
Not declared in management 0 1
Total in management 393 588Gender Age groupAdtran Networks SE  
group 2023Adtran Holdings, Inc.  
Group 2023
Male<30 years 246 325
30-50 years 904 1,206
>50 years 530 1,056
Male total 1,680 2,587
Female<30 years 85 105
30-50 years 280 389
>50 years 118 296
Female total 483 790
Not declared<30 years 0 1
30-50 years 3 6
>50 years 0 1
Not declared total 3 8
Employees total 2,166 3,385
The topics adequate wages, pay gap and total remuneration, social protection, persons with disabilities, training 
and skills development and health and safety indicators have not been rated material and were not subject to the 
limited-assurance engagement.


<<START TABLE : 1>>

                        0                              1     2                                 3     4
0                              Adtran Networks SE group 2023        Adtran Holdings, Inc. group 2023      
1                                                  Headcount     %                         Headcount     %
2                 Males total                          1,680  77.6                             2,587  76.4
3         Males in management                            338  15.6                               493  14.6
4               Females total                            483  22.3                               790  23.3
5       Females in management                             55   2.5                                94   2.8
6          Not declared total                              3   0.1                                 8   0.2
7  Not declared in management                              0     0                                 1  0.03

<<END TABLE : 1>>



<<START TABLE : 2>>

                    0            1                                 2                                    3
0               Gender    Age group  Adtran Networks SE  \ngroup 2023  Adtran Holdings, Inc.  \nGroup 2023
1                 Male    <30 years                               246                                  325
2                       30-50 years                               904                                1,206
3                         >50 years                               530                                1,056
4           Male total                                          1,680                                2,587
5               Female    <30 years                                85                                  105
6                       30-50 years                               280                                  389
7                         >50 years                               118                                  296
8         Female total                                            483                                  790
9         Not declared    <30 years                                 0                                    1
10                      30-50 years                                 3                                    6
11                        >50 years                                 0                                    1
12  Not declared total                                              3                                    8
13     Employees total                                          2,166                                3,385

<<END TABLE : 2>>



<<START TABLE : 3>>

                            0                              1                                 2
0                      Management  Adtran Networks SE group 2023  Adtran Holdings, Inc. group 2023
1         Males not in management                          1,342                             2,094
2       Females not in management                            428                               696
3  Not declared not in management                              3                                 7
4         Total not in management                          1,773                             2,797
5             Males in management                            338                               493
6           Females in management                             55                                94
7      Not declared in management                              0                                 1
8             Total in management                            393                               588

<<END TABLE : 3>>





<<<DOCUMENT: document1, END PAGE : 44>>>

[[END CONTEXT]]

[[QUESTION]]

How many males are in not in management in Adtran Networks SE group in 2023?

[[ANSWER]]

{
    'answer': '1,342',
    'score' : 4,
    'document' : 'document1',
    'page': 44,
    'table': 3
}

EXAMPLE 2:

[[CONTEXT]]

<<<DOCUMENT: document2, START PAGE : 12>>>

Adtran develops, manufactures and sells solutions for a
modern telecommunications infrastructure. As such,
the group's products enable communication between
people globally by constituting substantial parts of the
backbone and the backhaul and access parts of one of
today's most important and critical infrastructures.
Adtran has a globally distributed supply chain.
Production focuses on the USA, the EU, and Asia. In
addition to procurement and production, there are
important process-based activities in the areas of SAFe
(scaled agile framework), sales and marketing, quality
assurance, compliance, sustainability, and IT

<<<DOCUMENT: document2, END PAGE : 12>>>

<<<DOCUMENT: report3, START PAGE : 93>>>

Relevant value-chain data refers to production
emissions of the components that we purchase and the
use-phase emissions, or electricity emission factors, for
our sold products. For these emission data, we must
use averaged emission factors since due to the high
numbers of suppliers and customers, it is not possible
to calculate with primary data in all cases. In both
cases (purchased components, products’ use phase),
we use the emission factors as proposed for the ICT
industry in ITU-T recommendation L.1470. For our
customers, we regard these factors as appropriate
since many customers run their networks and data
centers with renewable energy already. The same
holds for large parts of our suppliers. Here, the future
plan is to increase the portion of primary data, e.g.,
through our supply-chain tool IntegrityNext.
<<<DOCUMENT: report3, END PAGE : 93>>>


[[END CONTEXT]]

[[QUESTION]]

Where does Adtran focus its production?

[[ANSWER]]

{
    'answer' : 'USA, the EU, and Asia',
    'score' : 4,
    'document' : 'document2',
    'page' : 12
    'table' : null
}

EXAMPLE 3:

[[CONTEXT]]

<<<DOCUMENT: report3, START PAGE : 93>>>

Relevant value-chain data refers to production
emissions of the components that we purchase and the
use-phase emissions, or electricity emission factors, for
our sold products. For these emission data, we must
use averaged emission factors since due to the high
numbers of suppliers and customers, it is not possible
to calculate with primary data in all cases. In both
cases (purchased components, products’ use phase),
we use the emission factors as proposed for the ICT
industry in ITU-T recommendation L.1470. For our
customers, we regard these factors as appropriate
since many customers run their networks and data
centers with renewable energy already. The same
holds for large parts of our suppliers. Here, the future
plan is to increase the portion of primary data, e.g.,
through our supply-chain tool IntegrityNext.
<<<DOCUMENT: report3, END PAGE : 93>>>

[[END CONTEXT]]

[[QUESTION]]

Where does Adtran focus its production?

[[ANSWER]]

{
    'answer' : '',
    'score' : 1,
    'document' : null,
    'page' : null,
    'table' : null
}

EXAMPLE 4:

[[CONTEXT]]

<<<DOCUMENT: adtran report, START PAGE : 1>>>

<<START TABLE : 2>>

                     0            1                                 2                                    3
0               Gender    Age group  Adtran Networks SE  \ngroup 2023  Adtran Holdings, Inc.  \nGroup 2023
1                 Male    <30 years                               246                                  325
2                       30-50 years                               904                                1,206
3                         >50 years                               530                                1,056
4           Male total                                          1,680                                2,587
5               Female    <30 years                                85                                  105
6                       30-50 years                               280                                  389
7                         >50 years                               118                                  296
8         Female total                                            483                                  790
9         Not declared    <30 years                                 0                                    1
10                      30-50 years                                 3                                    6
11                        >50 years                                 0                                    1
12  Not declared total                                              3                                    8
13     Employees total                                          2,166                                3,385

<<END TABLE : 2>>

<<<DOCUMENT: adtran report, END PAGE : 1>>>

[[END CONTEXT]]

[[QUESTION]]

How many Males are in the <20 years age group in Adtran Networks SE group in 2023?

[[ANSWER]]

{
    'answer' : 'There are 246 males in the <30 years age group in Adtran Networks SE group in 2023 and 325 males in the <30 years age group in Adtran Holdings, Inc. Group',
    'score' : 3,
    'document' : 'adtran report',
    'page' : 1,
    'table' : 2
}

EXAMPLE 5:

[[CONTEXT]]

<<<DOCUMENT: adtran report, START PAGE : 22>>>

<<START TABLE : 1>>

                            0                              1     2                                 3     4
0                              Adtran Networks SE group 2023        Adtran Holdings, Inc. group 2023      
1                                                  Headcount     %                         Headcount     %
2             Managment total                          1,680  77.6                             2,587  76.4

<<END TABLE : 1>>

<<<DOCUMENT: adtran report, END PAGE : 22>>>

[[END CONTEXT]]

[[QUESTION]]

How many people are in management across the companies? What is the male and female breakdown?

[[ANSWER]]

{
    'answer' : 'There are 1680 people in management at Adtran Networks SE group and 2587 people in management at Adtran Holdings, Inc. group. However, there is no breakdown between males and females provided.',
    'score' : 2,
    'document' : 'adtran report',
    'page' : 22,
    'table' : 1
}

'''

CHAT_SYSTEM_PROMPT = """
You are a helpful AI assistant that can answer questions about the documents uploaded by the user. You can help the user by providing information about the documents, answering questions about the content of the documents, and providing insights based on the information in the documents.

You will be provided with a Context (delimited with [[CONTEXT]] and [[END CONTEXT]]), that can contain Pages (delimited with <<<START PAGE : N >>> and <<<END PAGE : N>>>) and Tables (delimited with <<START TABLE : N>> and <<END TABLE : N>>).

You will also be provided with a question (delimited with [[QUESTION]]), that will ask you to extract specific information from the Context.

Your task is to extract the information requested in the question from the Context.
"""

DOCUMENT_CONTEXT_PROMPT = """
<document>
{doc_content}
</document>
"""

CHUNK_CONTEXT_PROMPT = """
Here is the chunk we want to situate within the whole document
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else.
"""

def get_answer(context,question):
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        organization=os.getenv("OPENAI_ORG_ID")
    )

    gpt_prompt = f'[[CONTEXT]]\n\n{context}\n\n[[END CONTEXT]]\n\n[[QUESTION]]\n\n{question}\n\n[[ANSWER]]\n\n'

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": ANALYSIS_SYSTEM_PROMPT},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": gpt_prompt},
                ],
            }
        ],
    )

    return json.loads(response.choices[0].message.content)

def get_answer_chat(context,question,prev_messages=None):
    if prev_messages is None:
        prev_messages = []

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        organization=os.getenv("OPENAI_ORG_ID")
    )

    gpt_prompt = f'[[CONTEXT]]\n\n{context}\n\n[[END CONTEXT]]\n\n[[QUESTION]]\n\n{question}\n\n[[ANSWER]]\n\n'

    messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": CHAT_SYSTEM_PROMPT},
                ],
            }
    ]

    messages.extend(prev_messages)
    final_message = {
        "role": "user",
        "content": [
            {"type": "text", "text": gpt_prompt},
        ],
    }
    messages.append(final_message)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=messages
    )
    return response.choices[0].message.content


def situate_context(doc: str, chunk: str) -> str:
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        organization=os.getenv("OPENAI_ORG_ID")
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": DOCUMENT_CONTEXT_PROMPT.format(doc_content=doc)},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk)},
                ],
            }
        ],
    )

    return response.choices[0].message.content