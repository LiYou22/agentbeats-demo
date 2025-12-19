import json

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart
from a2a.utils import get_message_text, new_agent_text_message

from eval_engine.run import main  # 你的 main 函数

class Agent:
    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)

        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Running evaluation...")
        )

        # input_text for main function persona
        result = main(
            persona=input_text,
            model="gpt-4o-mini",
            model_name=None
        )

        await updater.add_artifact(
            parts=[
                Part(
                    root=TextPart(
                        text=json.dumps(result, indent=2)
                    )
                )
            ],
            name="EvaluationResult"
        )
