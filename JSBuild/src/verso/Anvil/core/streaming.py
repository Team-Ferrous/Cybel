from cli.renderer import CLIRenderer


class StreamHandler:
    """
    Connects the LLM stream to the CLI renderer.
    """

    def __init__(self, renderer: CLIRenderer):
        self.renderer = renderer

    def stream_response(self, generator):
        """
        Consumes the LLM generator and renders output token-by-token.
        Returns the full accumulated response.
        """
        full_response = []
        for token in generator:
            self.renderer.stream_token(token)
            full_response.append(token)

        self.renderer.new_line()
        return "".join(full_response)
