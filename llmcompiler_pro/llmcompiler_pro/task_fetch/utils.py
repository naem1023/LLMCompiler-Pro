from .task import Task


def preprocess_instruction_from_task(task: Task):
    """Replace dependency placeholders, i.e. ${1}, in task.args with the actual observation.
    Not Implemented.
    """
    ...
