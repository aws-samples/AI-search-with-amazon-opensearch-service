import boto3
from botocore.exceptions import ClientError
import pprint
import time
import streamlit as st
from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
kendra_ranking = boto3.client("kendra-ranking",region_name = st.session_state.REGION)


print("Create a rescore execution plan.")

# Provide a name for the rescore execution plan
name = "MyRescoreExecutionPlan"
# Set your required additional capacity units
# Don't set capacity units if you don't require more than 1 unit given by default
capacity_units = 2

# try:
#     rescore_execution_plan_response = kendra_ranking.create_rescore_execution_plan(
#         Name = name,
#         CapacityUnits = {"RescoreCapacityUnits":capacity_units}
#     )

#     pprint.pprint(rescore_execution_plan_response)

#     rescore_execution_plan_id = rescore_execution_plan_response["Id"]

#     print("Wait for Amazon Kendra to create the rescore execution plan.")

#     while True:
#         # Get the details of the rescore execution plan, such as the status
#         rescore_execution_plan_description = kendra_ranking.describe_rescore_execution_plan(
#             Id = rescore_execution_plan_id
#         )
#         # When status is not CREATING quit.
#         status = rescore_execution_plan_description["Status"]
#         print(" Creating rescore execution plan. Status: "+status)
#         time.sleep(60)
#         if status != "CREATING":
#             break

# except ClientError as e:
#         print("%s" % e)

# print("Program ends.")



def re_rank(self_, rerank_type, search_type, question, answers):
    
    print("start")
    print()
    
        
    ans = []
    ids = []
    ques_ans = []
    query = question[0]['question']
    for i in answers[0]['answer']:
        if(self_ == "search"):
            
            ans.append({
                    "Id": i['id'],
                    "Body": i["desc"],
                    "OriginalScore": i['score'],
                    "Title":i["desc"]
                    })
            ids.append(i['id'])
            ques_ans.append((query,i["desc"]))
        
        else:
            ans.append({'text':i})
            
            ques_ans.append((query,i))
        
            

    re_ranked = [{}]

  

    

    if(rerank_type == 'Kendra Rescore'):


        

        rescore_response = kendra_ranking.rescore(
            RescoreExecutionPlanId = 'b2a4d4f3-98ff-4e17-8b69-4c61ed7d91eb',
            SearchQuery = query,
            Documents = ans
        )
    
            
        #[{'DocumentId': 'DocId1', 'Score': 2.0}, {'DocumentId': 'DocId2', 'Score': 1.0}]   
            
        
        re_ranked[0]['answer']=[]
        for result in rescore_response["ResultItems"]:

            pos_ = ids.index(result['DocumentId'])

            re_ranked[0]['answer'].append(answers[0]['answer'][pos_])
        re_ranked[0]['search_type']=search_type,
        re_ranked[0]['id'] = len(question)

        #st.session_state.answers_none_rank = st.session_state.answers
        return re_ranked
        

    # if(rerank_type == 'None'):
        
    #     st.session_state.answers = st.session_state.answers_none_rank 
        

    if(rerank_type == 'Cross Encoder'):

        scores = model.predict(
                    ques_ans
                        )
        
        print("scores")
        print(scores)
        index__ = 0
        for i in ans:
            i['new_score'] = scores[index__]
            index__ = index__+1

        ans_sorted = sorted(ans, key=lambda d: d['new_score'],reverse=True) 
        
        
        def retreive_only_text(item):
            return item['text']
            
        if(self_ == 'rag'):
            return list(map(retreive_only_text, ans_sorted)) 

       
        re_ranked[0]['answer']=[]
        for j in ans_sorted:
            pos_ = ids.index(j['Id'])
            re_ranked[0]['answer'].append(answers[0]['answer'][pos_])
        re_ranked[0]['search_type']= search_type,
        re_ranked[0]['id'] = len(question)
        return re_ranked




    #return st.session_state.answers
    




