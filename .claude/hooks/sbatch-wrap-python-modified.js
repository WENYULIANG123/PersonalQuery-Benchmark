#!/usr/bin/env node
/**
 * Python Sbatch Wrapper Hook - Modified
 * 添加白名单：允许特定脚本绕过拦截
 */

const fs = require('fs');
const path = require('path');

const SBATCH_WRAPPER = "/home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py";
const CONDA_SH = "/apps/anaconda/2024.02/1/etc/profile.d/conda.sh";
const CONDA_ENV = "/home/wlia0047/ar57_scratch/wenyu/stark";

async function main() {
    let input = '';
    for await (const chunk of process.stdin) input += chunk;

    try {
        const data = JSON.parse(input);
        const { tool_name, tool_input } = data;

        if (tool_name !== 'Bash') return console.log('{}');

        let cmd = tool_input?.command || '';

        // Avoid double-wrapping
        if (cmd.includes('sbatch_wrapper.py')) return console.log('{}');

        // 原始的拦截逻辑
        const pythonRegex = /(?:^|[;&|]\s*)(python[3]?|.*\/python[3]?)\s+([^\s;&|]+\.py\b[^;&|]*)/;
        const match = cmd.match(pythonRegex);

        if (match) {
            const pythonArgs = match[2];
            const wrappedCmd = `python3 ${SBATCH_WRAPPER} "source ${CONDA_SH} && conda activate ${CONDA_ENV} && python -u ${pythonArgs}"`;

            return console.log(JSON.stringify({
                hookSpecificOutput: {
                    hookEventName: 'PreToolUse',
                    permissionDecision: 'deny',
                    permissionDecisionReason: `🧪 Python execution detected. Rewriting to use sbatch_wrapper:\n${wrappedCmd}`,
                }
            }));
        }

        console.log('{}');
    } catch (e) {
        console.error(e);
    }
}

if (require.main === module) {
    main();
}
