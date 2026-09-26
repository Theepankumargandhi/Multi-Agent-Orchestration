from slugger import slugify


def test_collapses_adjacent_separators():
    assert slugify("Hello,   Agent World!") == "hello-agent-world"


def test_lowercase_words():
    assert slugify("AgentForge") == "agentforge"
