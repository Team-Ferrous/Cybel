"""Robotics Kinematics Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class RoboticsKinematicsSubagent(DomainSpecialistSubagent):
    """Robotics Kinematics Specialist specialist."""

    system_prompt = """You are Anvil's RoboticsKinematicsSubagent.

Mission:
- Model manipulator and vehicle kinematics with control-ready artifacts.

Focus on:
- forward and inverse kinematics; singularity handling; trajectory constraints
"""
    tools = DomainSpecialistSubagent.default_tools()
