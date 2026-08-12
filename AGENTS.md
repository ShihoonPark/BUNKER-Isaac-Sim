# Repository Guidelines

## Project Structure & Module Organization

This repository contains an Isaac Sim workspace for the AgileX BUNKER MINI 2.0 tracked robot.

- `scenes/` stores committed USD stages; `base_physics.usd` is the current physics test scene.
- `scripts/` contains launch and maintenance utilities, including `run_isaac.sh`.
- `assets/` is reserved for external robot models and generated imports. These large or third-party files are intentionally ignored.
- `logs/` contains runtime output such as `isaac_sim.log` and must not be committed.

Keep reusable scenes in `scenes/` and executable automation in `scripts/`. Do not commit ROS bags, point clouds, downloaded repositories, or generated Isaac Sim artifacts.

## Build, Test, and Development Commands

There is no separate build system yet. The expected environment is Ubuntu 22.04, Isaac Sim 5.1, and ROS 2 Humble.

```bash
./scripts/run_isaac.sh
tail -f logs/isaac_sim.log
```

The first command starts Isaac Sim in the background with the bundled Humble ROS 2 bridge configuration. The second follows startup and runtime diagnostics. To obtain the uncommitted AgileX models, run:

```bash
git clone https://github.com/agilexrobotics/ugv_gazebo_sim.git assets/ugv_gazebo_sim
```

## Coding Style & Naming Conventions

Write Bash scripts with a `#!/usr/bin/env bash` shebang, two-space indentation, quoted paths, and descriptive lowercase `snake_case` names. Prefer repository-relative paths over machine-specific absolute paths when adding new automation. Name scenes with lowercase `snake_case` (for example, `track_contact_test.usd`). Keep scripts focused and document environment variables that affect Isaac Sim or ROS.

## Testing Guidelines

Automated tests and coverage requirements are not yet configured. Validate launcher changes with `bash -n scripts/run_isaac.sh`, then start Isaac Sim and inspect `logs/isaac_sim.log`. For scene changes, open the USD stage and verify that it loads without errors, collisions behave as intended, and the ROS 2 Bridge remains available. Describe these manual checks in the pull request.

## Commit & Pull Request Guidelines

Recent history uses Conventional Commit-style subjects, such as `chore: initialize BUNKER Isaac Sim project`. Continue with short, imperative subjects using prefixes like `feat:`, `fix:`, `docs:`, or `chore:`. Pull requests should explain the simulation behavior changed, list validation steps, and link relevant issues. Include screenshots or short recordings for visible scene, physics, or UI changes, and call out any required external assets or local configuration.

## Agent-Specific Safety and Research Rules

- Work only inside this repository unless the user explicitly approves otherwise. Never use `sudo`, `apt`, system package installation, driver changes, global Git configuration, or permanent shell/environment changes.
- Treat `assets/ugv_gazebo_sim/` as read-only external upstream content. Treat `assets/bunker_mini_imported/` as read-only URDF-importer output; put derived assets or scripts elsewhere. Never alter third-party licensing or copyright information.
- Do not run `git commit`, `git push`, force-push, branch deletion, or destructive Git commands unless explicitly requested.
- Never add ROS bags, DB3/MCAP files, PCD datasets, runtime logs, caches, or other large generated data to Git.
- Target NVIDIA Isaac Sim 5.1.0 and ROS 2 Humble; verify APIs rather than assuming behavior from another Isaac Sim release.
- Do not invent BUNKER geometry, mass, inertia, track dimensions, contact parameters, or actuator parameters. Use measurements, source files, or manuals; otherwise label assumptions clearly.
- Document every tracked-vehicle physics simplification and the real behavior it may fail to reproduce.
- After code changes, run the safest relevant local checks. Report changed files, tests performed, remaining assumptions, and the resulting Git diff.
