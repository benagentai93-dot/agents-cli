# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""agents-cli setup command — install skills via npx skills CLI."""

import random
import shlex
from pathlib import Path

import click

from google.agents.cli._runner import run
from google.agents.cli._skills_check import SKILLS_NPX_PACKAGE
from google.agents.cli._tools import run_npx_skills

_MOTTOS = [
    "Give your coding agent the power to build ADK projects.",
    "From prototype to production — one CLI away.",
    "Skills up. Ship faster.",
    "Agents skills, installed in seconds.",
    "Your coding agent just got an upgrade.",
]

_DEFAULT_SKILLS_SOURCE = "https://github.com/google/agents-cli"


def _print_logo():
    """Print the AGENTS CLI ASCII art logo with a random motto."""
    click.secho(
        " █▀█ █▀▀ █▀▀ █▄ █ ▀█▀ █▀   █▀▀ █  █",
        fg="blue",
        bold=True,
    )
    click.secho(
        " █▀█ █▄█ ██▄ █ ▀█  █  ▄█   █▄▄ █▄ █",
        fg="cyan",
        bold=True,
    )
    click.echo()
    click.echo(f" {random.choice(_MOTTOS)}")


def _print_section(number, title):
    """Print a numbered section header."""
    click.echo()
    click.secho(f" {number}. {title}", bold=True)
    click.echo(f" {'─' * (len(title) + 3)}")


def _get_source_root():
    """Return the cwd if it is the agents-cli repo root, else None.

    Checks that a ``pyproject.toml`` with the ``agents-cli`` package
    name exists in the current working directory.
    """
    candidate = Path.cwd() / "pyproject.toml"
    if candidate.is_file():
        try:
            text = candidate.read_text()
            if 'name = "google-agents-cli"' in text:
                return Path.cwd()
        except OSError:
            pass
    return None


def _check_legacy_skills():
    """Check for legacy ADK skills and warn the user if found."""
    # ── Legacy Skills Detection ──
    try:
        import json
        import logging

        # Run raw command to get all skills without filtering
        result = run(
            ["npx", "-y", "skills", "list", "--json"],
            capture=True,
            check=False,
            timeout=15,
        )
        if result.returncode == 0:
            installed_skills = json.loads(result.stdout)
            legacy_skills = {
                "adk-cheatsheet",
                "adk-deploy-guide",
                "adk-dev-guide",
                "adk-eval-guide",
                "adk-observability-guide",
                "adk-scaffold",
            }

            found_legacy = []
            for skill in installed_skills:
                name = skill.get("name")
                if name in legacy_skills:
                    found_legacy.append(name)

            if found_legacy:
                click.secho(
                    f"\n⚠️  Warning: Found legacy ADK skills installed: {', '.join(found_legacy)}",
                    fg="yellow",
                )
                click.secho(
                    "     These may conflict with the new `agents-cli` skills.",
                    fg="yellow",
                )
                click.secho(
                    "     We suggest you uninstall them to avoid confusion, e.g.:",
                    dim=True,
                )
                for name in found_legacy:
                    click.secho(f"     npx skills remove {name}", dim=True)
                click.echo()
    except Exception as e:
        # Don't fail the setup if the check fails
        logging.warning("Could not check for legacy skills: %s", e)


@click.command("setup")
@click.option(
    "--workspace",
    is_flag=True,
    default=False,
    help=(
        "Install to project/workspace scope instead of global. "
        "Skills are installed relative to the current directory."
    ),
)
@click.option(
    "--skip-auth",
    is_flag=True,
    default=False,
    help="Skip the authentication step.",
)
@click.option(
    "--dry-run",
    "--dryrun",
    is_flag=True,
    default=False,
    help="Show what would be done without making changes.",
)
@click.option(
    "--dev",
    is_flag=True,
    default=False,
    help="Install as editable from the local repo (for contributors).",
)
@click.option(
    "--interactive",
    "-i",
    is_flag=True,
    default=False,
    help="Enable interactive authentication prompt if not already authenticated.",
)
@click.option(
    "--skills-source",
    default=None,
    help="Skills source: local path, GitHub owner/repo, or URL. Overrides the bundled skills.",
)
@click.option(
    "--agent",
    multiple=True,
    help=(
        "Specify the agent to install skills to (e.g. --agent claude-code --agent cursor). "
        "Use 'all' to install for all supported agents."
    ),
)
def cmd_setup(*, workspace, skip_auth, dry_run, dev, interactive, skills_source, agent):
    """Install agents-cli and skills to detected coding agents.

    Installs the agents-cli tool (via uv tool install) and detects
    installed coding agents (Claude Code, Antigravity CLI, Cursor,
    Windsurf, etc.) to install ADK development skills via npx skills.

    By default, skills are installed globally for all detected agents.
    Use --workspace to install at the project level instead.
    Use --agent to specify specific coding agents (e.g. --agent claude-code --agent cursor) or 'all'.
    Use --dry-run to preview what would happen without executing.
    Use --dev to install agents-cli as editable from the local repo (for contributors).
    Use --interactive / -i to enable interactive authentication if not already logged in.
    """
    click.echo("Setting up...")
    click.echo()
    _print_logo()

    scope = "workspace" if workspace else "global"

    source = _DEFAULT_SKILLS_SOURCE
    if dev and not skills_source:
        # In dev mode, use skills from the local repo checkout
        source = str(Path.cwd() / "skills")
    elif skills_source:
        source_path = Path(skills_source)
        # Only resolve to an absolute path if the source is a local file/directory
        # to prevent resolving remote URIs (e.g., GitHub URLs or package identifiers).
        if skills_source.startswith((".", "/")) or source_path.exists():
            source = str(source_path.resolve())
        else:
            source = skills_source
    args = ["add", source, "-y"]
    if "all" in agent:
        args.append("--all")
    elif agent:
        for a in agent:
            args.extend(["--agent", a])
    if not workspace:
        args.append("-g")

    # ── Dry Run ──
    if dry_run:
        _print_section(1, "Dry Run")
        click.echo()
        if dev:
            project_root = _get_source_root()
            if not project_root:
                raise click.ClickException(
                    "--dev requires running from the root of the agents-cli repository"
                )
            click.echo("  Would install agents-cli (editable):")
            click.secho(
                f"  \u25b8 uv tool install --force --editable {project_root}",
                fg="cyan",
                dim=True,
            )
        else:
            click.echo("  Would install agents-cli:")
            click.secho(
                "  \u25b8 uv tool install google-agents-cli",
                fg="cyan",
                dim=True,
            )
        click.echo()
        click.echo("  Would install skills:")
        full_args = ["npx", "-y", SKILLS_NPX_PACKAGE, *args]
        click.secho(f"  \u25b8 {shlex.join(full_args)}", fg="cyan", dim=True)
        click.echo(f"  Scope: {scope}")
        # Temporary compatibility step (see TODO at the real linking call below).
        if not workspace and (Path.home() / ".gemini").is_dir():
            click.echo(
                "  Would link global skills into Antigravity's skill directories "
                "(~/.gemini/config/skills, ~/.gemini/antigravity-cli/skills)."
            )
        click.echo()

        if not skip_auth:
            from google.agents.cli.auth import is_authenticated

            authed, display = is_authenticated()
            if authed:
                click.echo(f"  Auth:  {display}")
            else:
                click.echo("  Auth:  Not authenticated")
        else:
            click.echo("  Auth:  Skipped")

        click.echo()
        click.secho("  No changes made (dry run).", fg="yellow")
        click.echo()
        return

    step = 1

    # ── Authentication ──
    _print_section(step, "Authentication")
    step += 1
    if not skip_auth:
        from google.agents.cli.auth import is_authenticated, run_auth_step

        authed, display = is_authenticated()
        if authed:
            click.echo()
            click.secho(f"  Authenticated as {display}", fg="green")
        elif interactive:
            run_auth_step(show_header=False)
        else:
            click.echo()
            click.secho(
                "  Not authenticated. Run with --interactive (-i) to authenticate interactively.",
                fg="yellow",
                dim=True,
            )
    else:
        click.echo()
        click.secho("  Skipped (--skip-auth)", dim=True)

    # ── CLI Installation ──
    _print_section(step, "CLI Installation")
    step += 1
    click.echo()
    cli_installed = False
    if dev:
        project_root = _get_source_root()
        if not project_root:
            raise click.ClickException(
                "--dev requires running from the root of the agents-cli repository"
            )
        tool_args = [
            "uv",
            "tool",
            "install",
            "--force",
            "--editable",
            str(project_root),
        ]
    else:
        tool_args = ["uv", "tool", "install", "google-agents-cli"]
    result = run(tool_args, capture=True, check=False)
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    if "already installed" in stdout.lower() or "already installed" in stderr.lower():
        from google.agents.cli.scaffold.utils.version import check_for_updates

        needs_update, current, latest = check_for_updates()
        if needs_update:
            click.secho(
                f"  Installed ({current}), but {latest} is available.",
                fg="yellow",
            )
            click.secho(
                "  Run 'uv tool upgrade google-agents-cli' to update.",
                dim=True,
            )
        else:
            click.secho("  Already installed and up to date.", dim=True)
        cli_installed = True
    elif result.returncode != 0:
        click.secho("  Could not install agents-cli automatically.", fg="yellow")
        if stderr.strip():
            for line in stderr.strip().splitlines():
                click.echo(f"  {line}")
        click.secho("  Install manually: uv tool install google-agents-cli", dim=True)
    else:
        for line in stdout.strip().splitlines():
            click.echo(f"  {line}")
        cli_installed = True

    # ── Skills Installation ──
    _print_section(step, "Skills Installation")
    step += 1
    click.echo()

    # ── Legacy Skills Detection ──
    _check_legacy_skills()

    run_npx_skills(args, "Installing skills")

    # ── Antigravity skill links ──
    # Temporary until npx skills supports Antigravity's IDE / CLI / 2.0 paths:
    # npx installs global skills to ~/.agents/skills, which Antigravity does not
    # read, so mirror them into the locations the IDE/2.0 and CLI look in.
    # TODO(b/520131431): remove once Antigravity/npx align on skill paths.
    if not workspace:
        from google.agents.cli.setup._antigravity import link_skills_for_antigravity

        for line in link_skills_for_antigravity():
            click.echo(f"  {line}")

    # ── Summary ──
    _print_section(step, "Summary")
    click.echo()

    # Auth status
    if skip_auth:
        click.echo("  Auth:   Skipped")
    else:
        from google.agents.cli.auth import is_authenticated

        authed, display = is_authenticated()
        if authed:
            click.echo(f"  Auth:   {display}")
        else:
            click.echo("  Auth:   Not authenticated")

    # CLI tool status
    if cli_installed:
        if dev:
            click.echo("  CLI:    agents-cli installed (editable)")
        else:
            click.echo("  CLI:    agents-cli installed")
    else:
        click.echo("  CLI:    Not installed (run: uv tool install google-agents-cli)")

    # Skills status
    click.echo("  Skills: Installed")
    click.echo(f"  Scope:  {scope}")

    click.echo()
    if not cli_installed:
        raise click.ClickException("agents-cli installation failed")
    click.secho("  Done.", fg="green", bold=True)
    click.echo()
