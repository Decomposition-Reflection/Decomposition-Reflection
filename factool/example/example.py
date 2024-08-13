'''
Remember to export you API keys first.
export OPENAI_API_KEY='sk-SVYDJsZ21fL7Nt922a771aD74aBe4fB59a6776148bB16f44'
export SERPER_API_KEY='ebb2179c200d751dd3814fea91f37cf56598540f'
export SCRAPER_API_KEY=...
'''

from factool import Factool

factool_instance = Factool("gpt-3.5-turbo")

# one example input for each category.
inputs = [
            {
                "prompt": "How to identify the needs and expectations of each stakeholder in a marketing campaign?",
                "response": "In order to identify the needs and expectations of each stakeholder in a marketing campaign, it is important to conduct market research and gather feedback from stakeholders through surveys, focus groups, and other methods. This can help to identify their pain points, challenges, and goals, and tailor the marketing campaign to meet their specific needs and expectations. Additionally, it is important to regularly monitor and evaluate the success of the campaign, and make adjustments as needed based on stakeholder feedback.",
                "category": "kbqa"
            }
]

response_list = factool_instance.run(inputs)

print(response_list)