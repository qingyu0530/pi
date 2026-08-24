import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { StringEnum } from "@earendil-works/pi-ai";

// GitHub control extension.
// Requires the GitHub CLI (`gh`) to be installed and authenticated (`gh auth login`).
// All mutating actions are gated behind a confirmation dialog when a UI is available.
export default function (pi: ExtensionAPI) {
  async function checkGh(ctx: ExtensionContext): Promise<void> {
    const auth = await pi.exec("gh", ["auth", "status"], { signal: ctx.signal });
    if (auth.code !== 0) {
      throw new Error(
        "gh CLI is not authenticated. Run `gh auth login` first.\n" +
          (auth.stderr || auth.stdout),
      );
    }
  }

  async function isGitRepo(cwd: string, ctx: ExtensionContext): Promise<boolean> {
    const r = await pi.exec("git", ["rev-parse", "--is-inside-work-tree"], { cwd, signal: ctx.signal });
    return r.code === 0;
  }

  // Stage everything and commit. Returns a short status string.
  async function commitAll(ctx: ExtensionContext, message: string): Promise<string> {
    const cwd = ctx.cwd;
    const status = await pi.exec("git", ["status", "--porcelain"], { cwd, signal: ctx.signal });
    if (status.stdout.trim() === "") {
      return "No changes to commit.";
    }
    await pi.exec("git", ["add", "-A"], { cwd, signal: ctx.signal });
    const commit = await pi.exec("git", ["commit", "-m", message], { cwd, signal: ctx.signal });
    if (commit.code !== 0) {
      throw new Error(`git commit failed:\n${commit.stderr || commit.stdout}`);
    }
    return commit.stdout.trim() || "Committed.";
  }

  async function confirmOrThrow(ctx: ExtensionContext, title: string, body: string): Promise<void> {
    if (ctx.hasUI) {
      const ok = await ctx.ui.confirm(title, body);
      if (!ok) throw new Error("Cancelled by user.");
    }
  }

  pi.registerTool({
    name: "gh_create_repo",
    label: "Create GitHub Repo",
    description:
      "Create a new GitHub repository for the current project and push the code to it. " +
      "Requires the gh CLI to be logged in.",
    promptSnippet: "Create a GitHub repository and push the current project to it",
    parameters: Type.Object({
      name: Type.String({ description: "Repository name, e.g. my-project" }),
      visibility: StringEnum(["public", "private"] as const, {
        description: "Repository visibility",
      }),
      description: Type.Optional(Type.String({ description: "Optional repository description" })),
    }),
    async execute(_toolCallId, params, signal, _onUpdate, ctx) {
      await confirmOrThrow(
        ctx,
        "Create GitHub repo?",
        `Create "${params.name}" (${params.visibility}) and push the current code?`,
      );
      await checkGh(ctx);

      const cwd = ctx.cwd;
      if (!(await isGitRepo(cwd, ctx))) {
        await pi.exec("git", ["init", "-b", "main"], { cwd, signal });
      }
      await commitAll(ctx, "Initial commit");

      const args = ["repo", "create", params.name, `--${params.visibility}`];
      if (params.description) args.push("--description", params.description);
      args.push("--source", ".", "--push");

      const res = await pi.exec("gh", args, { cwd, signal });
      if (res.code !== 0) {
        throw new Error(`gh repo create failed:\n${res.stderr || res.stdout}`);
      }
      return {
        content: [{ type: "text", text: `Created and pushed to ${params.name}.` }],
        details: { name: params.name, visibility: params.visibility },
      };
    },
  });

  pi.registerTool({
    name: "gh_push",
    label: "Push to GitHub",
    description:
      "Commit all changes and push to the current git remote. Requires an existing remote (origin).",
    promptSnippet: "Commit all changes and push to the GitHub remote",
    parameters: Type.Object({
      commitMessage: Type.String({ description: "Commit message" }),
      branch: Type.Optional(Type.String({ description: "Branch to push (default: current)" })),
      remote: Type.Optional(Type.String({ description: "Remote name (default: origin)" })),
      force: Type.Optional(Type.Boolean({ description: "Force push (dangerous, overwrites remote)" })),
    }),
    async execute(_toolCallId, params, signal, _onUpdate, ctx) {
      if (params.force) {
        await confirmOrThrow(ctx, "Force push?", "Force push will overwrite remote history. Continue?");
      }
      const msg = await commitAll(ctx, params.commitMessage);

      const cwd = ctx.cwd;
      const remote = params.remote ?? "origin";
      const pushArgs = ["push"];
      if (params.force) pushArgs.push("--force");
      pushArgs.push(remote);
      if (params.branch) pushArgs.push(params.branch);

      const res = await pi.exec("git", pushArgs, { cwd, signal });
      if (res.code !== 0) {
        throw new Error(`git push failed:\n${res.stderr || res.stdout}`);
      }
      const target = `${remote}${params.branch ? "/" + params.branch : ""}`;
      return {
        content: [{ type: "text", text: `${msg}\nPushed to ${target}.` }],
        details: {},
      };
    },
  });

  pi.registerTool({
    name: "gh_create_issue",
    label: "Create GitHub Issue",
    description: "Create a GitHub issue in the current repository via gh.",
    promptSnippet: "Create a GitHub issue in the current repository",
    parameters: Type.Object({
      title: Type.String({ description: "Issue title" }),
      body: Type.Optional(Type.String({ description: "Issue body / description" })),
    }),
    async execute(_toolCallId, params, signal, _onUpdate, ctx) {
      await confirmOrThrow(ctx, "Create issue?", `Create issue "${params.title}"?`);
      await checkGh(ctx);

      const args = ["issue", "create", "--title", params.title];
      if (params.body) args.push("--body", params.body);
      const res = await pi.exec("gh", args, { cwd: ctx.cwd, signal });
      if (res.code !== 0) {
        throw new Error(`gh issue create failed:\n${res.stderr || res.stdout}`);
      }
      return {
        content: [{ type: "text", text: res.stdout.trim() || "Issue created." }],
        details: {},
      };
    },
  });

  pi.registerTool({
    name: "gh_open_pr",
    label: "Open GitHub PR",
    description: "Open a pull request from the current branch to the base branch via gh.",
    promptSnippet: "Open a pull request from the current branch",
    parameters: Type.Object({
      title: Type.String({ description: "PR title" }),
      body: Type.Optional(Type.String({ description: "PR description" })),
      base: Type.Optional(Type.String({ description: "Base branch (default: the repo default)" })),
    }),
    async execute(_toolCallId, params, signal, _onUpdate, ctx) {
      await confirmOrThrow(ctx, "Open PR?", `Open pull request "${params.title}"?`);
      await checkGh(ctx);

      const args = ["pr", "create", "--title", params.title];
      if (params.body) args.push("--body", params.body);
      if (params.base) args.push("--base", params.base);
      const res = await pi.exec("gh", args, { cwd: ctx.cwd, signal });
      if (res.code !== 0) {
        throw new Error(`gh pr create failed:\n${res.stderr || res.stdout}`);
      }
      return {
        content: [{ type: "text", text: res.stdout.trim() || "PR created." }],
        details: {},
      };
    },
  });
}
