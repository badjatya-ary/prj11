
#!/usr/bin/env python3
"""
Python Automation Script with OpenAI API Integration
Automatically enhances user prompts and generates executable code.
"""

import os
import sys
import re
import time
import tempfile
import subprocess
from typing import Optional, List, Tuple

from dotenv import load_dotenv
load_dotenv()

# Import required packages for OpenAI and LangChain
try:
    from openai import OpenAI
    from langchain_openai import ChatOpenAI
    from langchain.prompts import PromptTemplate
except ImportError as e:
    print(f"Error: Required packages not installed: {e}")
    print("Please install required packages:")
    print("pip install openai langchain-openai langchain-core")
    sys.exit(1)


class OpenAIAutomationScript:
    """Main class for the OpenAI automation script."""

    def __init__(self):
        """Initialize the script with OpenAI clients."""
        self.openai_client = None
        self.langchain_client = None
        self.max_install_retries = 3
        self.max_api_retries = 3

        # Initialize OpenAI clients
        self._initialize_clients()

        # Create system prompt for GPT-3.5 to enhance user prompts
        self.enhancement_prompt = PromptTemplate(
            input_variables=["user_task"],
            template="""You are a prompt enhancement expert. Your task is to rewrite the user's task description into a comprehensive, structured, and detailed prompt that will guide GPT-4 to generate complete, executable Python code without ambiguity or incompleteness.

The enhanced prompt should:
1. Be extremely specific about requirements
2. Request complete, executable code with all necessary imports
3. Include error handling where appropriate
4. Specify exact output format and behavior
5. Request comprehensive documentation and comments
6. Ensure no truncation or incomplete implementations

User's original task: {user_task}

Enhanced prompt for GPT-4:"""
        )

    def _initialize_clients(self):
        """Initialize OpenAI and LangChain clients."""
        try:
            # Check for API key
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not found")

            # Initialize OpenAI client
            self.openai_client = OpenAI(api_key=api_key)

            # Initialize LangChain client for GPT-3.5
            self.langchain_client = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.1,
                max_tokens=2000,
                openai_api_key=api_key
            )

            print("✅ OpenAI clients initialized successfully")

        except Exception as e:
            print(f"❌ Failed to initialize OpenAI clients: {e}")
            sys.exit(1)

    def get_user_input(self) -> str:
        """Prompt user for task description."""
        print("\n" + "="*60)
        print("🤖 AI Code Generator - Python Automation Script")
        print("="*60)
        print("Please describe the task you want to accomplish:")
        print("(Type your description and press Enter)")
        print("-" * 60)

        user_input = input("> ").strip()

        if not user_input:
            print("❌ Error: Empty input provided")
            return self.get_user_input()

        print(f"✅ User input received: {len(user_input)} characters \n {user_input}")
        return user_input

    def enhance_prompt_with_gpt35(self, user_task: str) -> str:
        """Use GPT-3.5 via LangChain to enhance the user's prompt."""
        print("🔄 Enhancing prompt with GPT-3.5...")

        for attempt in range(self.max_api_retries):
            try:
                # Format the enhancement prompt
                formatted_prompt = self.enhancement_prompt.format(user_task=user_task)

                # Call GPT-3.5 via LangChain
                response = self.langchain_client.invoke(formatted_prompt)

                enhanced_prompt = response.content.strip()

                if enhanced_prompt:
                    print(f"✅ Prompt enhanced successfully ({len(enhanced_prompt)} characters) \n {enhanced_prompt}")
                    return enhanced_prompt
                else:
                    raise ValueError("Empty response from GPT-3.5")

            except Exception as e:
                print(f"⚠️  Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_api_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    print("❌ Failed to enhance prompt after all retries")
                    raise
        print(enhanced_prompt)
    def generate_code_with_gpt4(self, enhanced_prompt: str) -> str:
        """Use GPT-4 to generate Python code."""
        print("🔄 Generating code with GPT-4...")

        # Create system message for code generation
        system_message = """You are an expert Python developer. Generate complete, executable Python code based on the user's requirements. 

Requirements:
- Provide ONLY the Python code, no explanations
- Include all necessary imports
- Add comprehensive error handling
- Include docstrings and comments
- Ensure code is complete and executable
- Do not truncate or abbreviate any part of the implementation
 Please use the current python environment. Get the environment name and run the generated code within that envrironment.
 If a browser is mentioned please use that else get the default browser of the system and use that.
 If user requests any task related to OS manipulation use pyautogui module of python.
 if user says play youtube video use youtube-search-python library to fetch the link of the youtube video and then write code to run that link in the browser
 

"""

        for attempt in range(self.max_api_retries):
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": enhanced_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=5000
                )

                generated_code = response.choices[0].message.content.strip()

                if generated_code:
                    print(f"✅ Code generated successfully ({len(generated_code)} characters)")
                    return generated_code
                else:
                    raise ValueError("Empty response from GPT-4")

            except Exception as e:
                print(f"⚠️  Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_api_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    print("❌ Failed to generate code after all retries")
                    raise

    def extract_python_code(self, response_text: str) -> str:
        """Extract Python code from GPT response."""
        # Try to extract code from markdown code blocks
        code_block_pattern = r'```(?:python|py)?\n(.*?)\n```'
        matches = re.findall(code_block_pattern, response_text, re.DOTALL)

        if matches:
            # Return the first (and likely only) code block
            return matches[0].strip()

        # If no code blocks found, return the entire response
        # (assuming GPT-4 followed instructions to return only code)
        return response_text.strip()

    def create_temp_file(self, code: str) -> str:
        """Create a temporary Python file with the generated code."""
        print("📝 Creating temporary Python file...")

        try:
            # Extract clean Python code
            clean_code = self.extract_python_code(code)

            # Create temporary file
            with tempfile.NamedTemporaryFile(
                mode='w', 
                suffix='.py', 
                delete=False,
                encoding='utf-8'
            ) as temp_file:
                temp_file.write(clean_code)
                temp_file_path = temp_file.name

            print(f"✅ Temporary file created: {temp_file_path}")
            return temp_file_path

        except Exception as e:
            print(f"❌ Failed to create temporary file: {e}")
            raise

    def parse_missing_modules(self, error_message: str) -> List[str]:
        """Parse missing module names from error messages."""
        missing_modules = []

        # Pattern for ModuleNotFoundError
        module_pattern = r"No module named '([^']+)'"
        matches = re.findall(module_pattern, error_message)
        missing_modules.extend(matches)

        # Pattern for ImportError
        import_pattern = r"cannot import name '[^']+' from '([^']+)'"
        matches = re.findall(import_pattern, error_message)
        missing_modules.extend(matches)

        # Additional pattern for direct import errors
        direct_pattern = r"ModuleNotFoundError: No module named '([^']+)'"
        matches = re.findall(direct_pattern, error_message)
        missing_modules.extend(matches)

        # Remove duplicates and return
        return list(set(missing_modules))

    def install_package(self, package_name: str) -> bool:
        """Install a Python package using pip."""
        print(f"📦 Installing package: {package_name}")

        try:
            # Use subprocess to install package
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', package_name],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode == 0:
                print(f"✅ Successfully installed: {package_name}")
                return True
            else:
                print(f"❌ Failed to install {package_name}: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            print(f"⏰ Installation timeout for package: {package_name}")
            return False
        except Exception as e:
            print(f"❌ Installation error for {package_name}: {e}")
            return False

    def execute_code_file(self, file_path: str) -> Tuple[bool, str]:
        """Execute the Python file and handle missing dependencies."""
        print("🚀 Executing generated code...")

        install_attempts = {}  # Track installation attempts per package

        for execution_attempt in range(self.max_install_retries + 1):
            try:
                # Execute the Python file
                result = subprocess.run(
                    [sys.executable, file_path],
                    capture_output=True,
                    text=True,
                    timeout=6000000  # 1 minute timeout for execution
                )

                if result.returncode == 0:
                    print("✅ Code executed successfully!")
                    return True, result.stdout
                else:
                    error_output = result.stderr
                    print(f"⚠️  Execution failed with return code {result.returncode}")

                    # Check if it's a missing module error
                    missing_modules = self.parse_missing_modules(error_output)

                    if missing_modules and execution_attempt < self.max_install_retries:
                        print(f"🔍 Detected missing modules: {missing_modules}")

                        # Try to install missing modules
                        all_installed = True
                        for module in missing_modules:
                            # Skip if we've already tried to install this module too many times
                            if install_attempts.get(module, 0) >= self.max_install_retries:
                                print(f"⚠️  Skipping {module} (max install attempts reached)")
                                all_installed = False
                                continue

                            # Attempt to install the module
                            install_attempts[module] = install_attempts.get(module, 0) + 1
                            if not self.install_package(module):
                                all_installed = False

                        if all_installed:
                            print("🔄 Retrying code execution...")
                            continue

                    # If we reach here, execution failed and we can't fix it
                    return False, error_output

            except subprocess.TimeoutExpired:
                error_msg = "Code execution timed out (60 seconds)"
                print(f"⏰ {error_msg}")
                return False, error_msg
            except Exception as e:
                error_msg = f"Execution error: {e}"
                print(f"❌ {error_msg}")
                return False, error_msg

        return False, "Maximum installation attempts reached"

    def cleanup_temp_file(self, file_path: str):
        """Clean up the temporary file."""
        try:
            os.unlink(file_path)
            print(f"🗑️  Cleaned up temporary file: {file_path}")
        except Exception as e:
            print(f"⚠️  Warning: Could not clean up temporary file: {e}")

    def run(self):
        """Main execution flow."""
        temp_file_path = None

        try:
            # Step 1: Get user input
            user_task = self.get_user_input()

            # Step 2: Enhance prompt with GPT-3.5
            enhanced_prompt = self.enhance_prompt_with_gpt35(user_task)

            # Step 3: Generate code with GPT-4
            generated_code = self.generate_code_with_gpt4(enhanced_prompt)

            # Step 4: Create temporary file
            temp_file_path = self.create_temp_file(generated_code)

            # Step 5: Execute code with dependency management
            success, output = self.execute_code_file(temp_file_path)

            # Step 6: Display results
            print("\n" + "="*60)
            print("📊 EXECUTION RESULTS")
            print("="*60)

            if success:
                print("🎉 Status: SUCCESS")
                print("\n📄 Output:")
                if output.strip():
                    print(output)
                else:
                    print("(No output produced)")
            else:
                print("❌ Status: FAILED")
                print("\n🐛 Error Details:")
                print(output)

            print("="*60)

        except KeyboardInterrupt:
            print("\n\n⚠️  Operation cancelled by user")
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
        finally:
            # Cleanup
            if temp_file_path:
                pass
                # self.cleanup_temp_file(temp_file_path)


def main():
    """Entry point for the script."""
    try:
        script = OpenAIAutomationScript()
        while True:
            script.run()
            user_data = input("Please enter something to continue: ")
            print(user_data)
    except Exception as e:
        print(f"❌ Critical error: {e}")
        # sys.exit(1)


if __name__ == "__main__":
    main()
