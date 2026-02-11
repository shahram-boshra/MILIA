# Scientist creates this - often just empty
# OR optionally imports helpers if they exist:

# Attempt to import helpers, but don't fail if they don't exist
try:
    from .helpers import some_utility_function
except (ImportError, ModuleNotFoundError):
    # helpers module doesn't exist yet - that's fine
    # User can create it later if needed
    some_utility_function = None
