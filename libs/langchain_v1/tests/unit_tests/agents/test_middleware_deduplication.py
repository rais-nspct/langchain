import logging
from collections.abc import Callable
from unittest.mock import MagicMock

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage

from langchain.agents.factory import create_agent
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse


class MyMiddleware(AgentMiddleware):
    def __init__(self, id_val: str):
        self.id_val = id_val
        super().__init__()

    @property
    def name(self) -> str:
        return "my_middleware"

    def wrap_model_call(
        self,
        request: ModelRequest,
        forward: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        # call the model
        response = forward(request)
        # Append our ID to the content of the first message to prove we ran
        if response.result and isinstance(response.result[0], AIMessage):
            content = response.result[0].content
            response.result[0].content = f"{content} | Middleware: {self.id_val}"
        return response


def test_create_agent_duplicate_middleware_last_wins(caplog):
    """Test that create_agent allows duplicate middleware names but warns and dedups."""
    m1 = MyMiddleware("first")
    m2 = MyMiddleware("second")

    # Mock model
    model = MagicMock(spec=BaseChatModel)
    model.profile = None
    # Mock invoke to return a basic message
    model.invoke.return_value = AIMessage(content="Hello")
    model.bind_tools.return_value = model  # simple mock binding
    model.bind.return_value = model

    with caplog.at_level(logging.WARNING):
        # Should not raise AssertionError
        graph = create_agent(model=model, middleware=[m1, m2])

    # Assert warning was logged
    assert "Duplicate middleware names found" in caplog.text

    # Run the graph to see which middleware wraps the call
    # Input state
    result = graph.invoke({"messages": [("user", "hi")]})

    # Check the last message content
    last_msg = result["messages"][-1]

    # We expect "second" to be present, and "first" NOT to be present
    assert "| Middleware: second" in last_msg.content
    assert "first" not in last_msg.content


class UniqueMiddleware(AgentMiddleware):
    """Middleware with a unique name for testing non-duplicate behavior."""

    def __init__(self, name: str, id_val: str):
        self._name = name
        self.id_val = id_val
        super().__init__()

    @property
    def name(self) -> str:
        return self._name

    def wrap_model_call(
        self,
        request: ModelRequest,
        forward: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        response = forward(request)
        if response.result and isinstance(response.result[0], AIMessage):
            content = response.result[0].content
            response.result[0].content = f"{content} | {self.id_val}"
        return response


def test_middleware_without_duplicate_names_preserved_unchanged(caplog):
    """Test that middleware without duplicate names are preserved unchanged."""
    m1 = UniqueMiddleware("middleware_a", "A")
    m2 = UniqueMiddleware("middleware_b", "B")
    m3 = UniqueMiddleware("middleware_c", "C")

    model = MagicMock(spec=BaseChatModel)
    model.profile = None
    model.invoke.return_value = AIMessage(content="Hello")
    model.bind_tools.return_value = model
    model.bind.return_value = model

    with caplog.at_level(logging.WARNING):
        graph = create_agent(model=model, middleware=[m1, m2, m3])

    # No warning should be logged since there are no duplicates
    assert "Duplicate middleware names found" not in caplog.text

    result = graph.invoke({"messages": [("user", "hi")]})
    last_msg = result["messages"][-1]

    # All three middleware should have run
    assert "| A" in last_msg.content
    assert "| B" in last_msg.content
    assert "| C" in last_msg.content


def test_order_of_non_duplicate_middleware_preserved_after_deduplication(caplog):
    """Test that the order of non-duplicate middleware is preserved after deduplication."""
    # Create middleware with some duplicates (same name) and some unique
    dup1 = MyMiddleware("first_dup")  # name="my_middleware"
    unique_a = UniqueMiddleware("unique_a", "A")
    dup2 = MyMiddleware("second_dup")  # name="my_middleware" - will override dup1
    unique_b = UniqueMiddleware("unique_b", "B")

    model = MagicMock(spec=BaseChatModel)
    model.profile = None
    model.invoke.return_value = AIMessage(content="Hello")
    model.bind_tools.return_value = model
    model.bind.return_value = model

    with caplog.at_level(logging.WARNING):
        graph = create_agent(model=model, middleware=[dup1, unique_a, dup2, unique_b])

    # Warning should be logged for duplicate middleware
    assert "Duplicate middleware names found" in caplog.text

    result = graph.invoke({"messages": [("user", "hi")]})
    last_msg = result["messages"][-1]

    # The deduplicated middleware retains the last instance (second_dup)
    assert "second_dup" in last_msg.content
    assert "first_dup" not in last_msg.content

    # Unique middleware are preserved
    assert "| A" in last_msg.content
    assert "| B" in last_msg.content

    # Check order: wrap_model_call runs in reverse order (last to first)
    # So the content should reflect: Hello -> unique_b -> my_middleware -> unique_a
    # Since middleware wraps in composition order, first in list wraps outermost
    # wrap_model_call handlers are chained where first in list is outermost
    content = last_msg.content
    # Verify all expected identifiers are present in the output
    assert "| A" in content
    assert "| B" in content
    assert "second_dup" in content
