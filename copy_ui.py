import os
import shutil

src = r"d:\scaler\prompt-opt-env\web_app.py"
dst = r"d:\scaler\prompt-opt-env\prompt_opt_env\web_ui.py"

with open(src, 'r', encoding='utf-8') as f:
    content = f.read()

# Make imports robust for the Docker container where prompt_opt_env is the root
old_import = """from prompt_opt_env.server.actions import (
    count_tokens,
    add_context, shorten, add_example, rephrase, add_constraint
)"""

new_import = """try:
    from prompt_opt_env.server.actions import (
        count_tokens, add_context, shorten, add_example, rephrase, add_constraint
    )
except (ImportError, ModuleNotFoundError):
    from server.actions import (
        count_tokens, add_context, shorten, add_example, rephrase, add_constraint
    )
"""

content = content.replace(old_import, new_import)

with open(dst, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Successfully copied and patched UI to {dst}")
