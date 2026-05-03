import subprocess
import tempfile
import yaml
import os
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import ClassVar


# ---- CONFIG ----
LLM = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key="Enter Your API KEY",
    temperature=0.1,
    max_tokens=2048
)

# Helper: Generate Deployment YAML
def generate_deployment_yaml(name: str, image: str, replicas: int = 1, namespace: str = "default", port: int = 80):
    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": name, "namespace": namespace},
        "spec": {
            "replicas": replicas,
            "selector": {"matchLabels": {"app": name}},
            "template": {
                "metadata": {"labels": {"app": name}},
                "spec": {
                    "containers": [{
                        "name": name,
                        "image": image,
                        "ports": [{"containerPort": port}]
                    }]
                }
            }
        }
    }
    return yaml.dump(deployment, default_flow_style=False)

# Helper: Apply YAML with kubectl
def apply_yaml(yaml_content: str):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        temp_file = f.name
    cmd = ["kubectl", "apply", "-f", temp_file]
    result = subprocess.run(cmd, capture_output=True, text=True)
    os.unlink(temp_file)
    return result.stdout if result.returncode == 0 else result.stderr

# ✅ FIXED Tool: Create Deployment
class CreateDeploymentTool(BaseTool):
    name: str = "create_deployment"
    description: str = "Create a Kubernetes deployment. Input: name (str), image (str), replicas (int, default 1)."

    def _run(self, tool_input: str):
        """Parse the tool input and create deployment"""
        import json
        import re
        
        try:
            # Try to parse as JSON first
            data = json.loads(tool_input)
            name = data.get('name')
            image = data.get('image')
            replicas = data.get('replicas', 1)
        except json.JSONDecodeError:
            # Try to parse key=value format (e.g., "name=lw, image=nginx")
            if '=' in tool_input:
                # Extract name and image from key=value format
                name_match = re.search(r'name=([^,\s]+)', tool_input)
                image_match = re.search(r'image=([^,\s]+)', tool_input)
                replicas_match = re.search(r'replicas=(\d+)', tool_input)
                
                if name_match and image_match:
                    name = name_match.group(1)
                    image = image_match.group(1)
                    replicas = int(replicas_match.group(1)) if replicas_match else 1
                else:
                    raise ValueError("Could not parse name and image from input")
            else:
                # Try space-separated values
                parts = tool_input.split()
                if len(parts) >= 2:
                    name = parts[0]
                    image = parts[1]
                    replicas = int(parts[2]) if len(parts) > 2 else 1
                else:
                    raise ValueError("Invalid input format. Expected: name image [replicas] or JSON or name=value, image=value")
        
        yaml_content = generate_deployment_yaml(name, image, replicas)
        return apply_yaml(yaml_content)

    def _arun(self, *args, **kwargs):
        raise NotImplementedError()

# Build Agent
tools = [CreateDeploymentTool()]

# Use a simpler approach with default ReAct agent
from langchain.agents import initialize_agent, AgentType

agent = initialize_agent(
    tools, 
    LLM, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

# Run
if __name__ == "__main__":
    print("🤖 Simple Kubernetes AI Agent (Deployments only)")
    print("Using gemini-2.5-pro model")
    
    while True:
        try:
            user_input = input("\n💡 What should I do? (or 'exit'): ").strip()
            if user_input.lower() in ["exit", "quit"]:
                break
            result = agent.run(user_input)
            print("Agent:", result)
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please try again or type 'exit' to quit")
