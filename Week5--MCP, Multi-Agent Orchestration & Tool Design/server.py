
import json
from typing import Literal

from mcp.server import MCPServer#importing the class


mcp = MCPServer(name= "task-manager") #creating object

TASKS = [
    {"id": 1, "title": "Buy books", "priority": "high", "done": False},
    {"id": 2, "title": "Read books", "priority": "medium", "done": False},
    {"id": 3, "title": "Water the plants", "priority": "low", "done": True},
]

@mcp.resource(
    uri= "tasks://all",
    mime_type="application/json",
)
def all_tasks() -> str:
    """The complete to-do list, including tasks already marked done.
    Each task has an id, title, priority (low/medium/high) and a done flag."""
    return json.dumps(TASKS, indent=2)


@mcp.tool()
def add_task(title: str, priority: Literal["low", "medium", "high"] = "medium") -> str:
    """Create a new task on the to-do list.

    Use this whenever the user wants to add, create, record or remember a task.
    The task is created as not-done. Returns a confirmation with the new task id.

    Args:
        title: What the task is, e.g. "Buy milk".
        priority: How urgent the task is. Defaults to "medium" if unspecified.
    """
    new_id = max((task["id"] for task in TASKS), default=0) + 1

    task = {"id": new_id, "title": title, "priority": priority, "done": False}
    TASKS.append(task)

    return f"Added task #{new_id}: {title!r} (priority: {priority})"


if __name__ == "__main__":
    mcp.run()
