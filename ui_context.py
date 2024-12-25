import gradio as gr

from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI


model = ChatOpenAI(model='gpt-4o-mini')

def chat(system_prompt, user_prompt, temperature = 0, verbose = False):
    ''' Normal call of OpenAI API '''
    response = model(
    temperature = temperature,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])
    
    res = response.content
    
    if verbose:
        print('System prompt:', system_prompt)
        print('User prompt:', user_prompt)
        print('GPT response:', res)
        
    return res

context = '''Background
Higher resistance rates of > 20% have been noted in Enterobacteriaceae urinary isolates towards ciprofloxacin and co-trimoxazole (C + C) in Singapore, compared with amoxicillin-clavulanate and nitrofurantoin (AC + N). This study examined if treatment failure varied between different antibiotics, given different resistant rates, for uncomplicated urinary tract infections (UTIs) managed in primary care. We also aimed to identify gaps for improvement in diagnosis, investigations, and management.

Methods
A retrospective cohort study was conducted from 2019 to 2021 on female patients aged 18–50 with uncomplicated UTIs at 6 primary care clinics in Singapore. ORENUC classification was used to exclude complicated UTIs. Patients with uncomplicated UTIs empirically treated with amoxicillin-clavulanate, nitrofurantoin, ciprofloxacin or co-trimoxazole were followed-up for 28 days. Treatment failure was defined as re-attendance for symptoms and antibiotic re-prescription, or hospitalisation for UTI complications. After 2:1 propensity score matching in each group, modified Poisson regression and Cox proportional hazard regression accounting for matched data were used to determine risk and time to treatment failure.

Results
3194 of 4253 (75.1%) UTIs seen were uncomplicated, of which only 26% were diagnosed clinically. Urine cultures were conducted for 1094 (34.3%) uncomplicated UTIs, of which only 410 (37.5%) had bacterial growth. The most common organism found to cause uncomplicated UTIs was Escherichia coli (64.6%), with 92.6% and 99.4% of isolates sensitive to amoxicillin-clavulanate and nitrofurantoin respectively. Treatment failure occurred in 146 patients (4.57%). Among 1894 patients treated with AC + N matched to 947 patients treated with C + C, patients treated with C + C were 50% more likely to fail treatment (RR 1.49, 95% CI 1.10–2.01), with significantly higher risk of experiencing shorter time to failure (HR 1.61, 95% CI 1.12–2.33), compared to patients treated with AC + N.

Conclusion
Treatment failure rate was lower for antibiotics with lower reported resistance rates (AC + N). We recommend treating uncomplicated UTIs in Singapore with amoxicillin-clavulanate or nitrofurantoin, based on current local antibiograms. Diagnosis, investigations and management of UTIs remained sub-optimal. Future studies should be based on updating antibiograms, highlighting its importance in guideline development.'''

def format_chat_prompt(message, chat_history, max_convo_length):
    prompt = ""
    for turn in chat_history[-max_convo_length:]:
        user_message, bot_message = turn
        prompt = f"{prompt}\nUser: {user_message}\nAssistant: {bot_message}"
    prompt = f"{prompt}\nUser: {message}\nAssistant:"
    return prompt

def respond(message, chat_history, max_convo_length = 10):
        formatted_prompt = format_chat_prompt(message, chat_history, max_convo_length)
        print('Prompt + Context:')
        print(formatted_prompt)
        bot_message = chat(system_prompt = f'''You are a friendly chatbot. Generate the output for only the Assistant. 
Only answer based on the context. If unsure, output "I don't know". Context: {context}"''',
                           user_prompt = formatted_prompt)
        chat_history.append((message, bot_message))
        return "", chat_history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height=300) #just to fit the notebook
    msg = gr.Textbox(label="Prompt")
    btn = gr.Button("Submit")
    clear = gr.ClearButton(components=[msg, chatbot], value="Clear console")

    btn.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
    msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot]) #Press enter to submit
gr.close_all()
demo.launch(share=True)