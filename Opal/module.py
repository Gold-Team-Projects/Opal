from context import Context

class Module:
    activators = []
    name = ""
    
    def process(ctx: Context):
        ctx.data = ""