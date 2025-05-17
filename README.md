# FINCON: A Synthesized LLM Multi-Agent System for Enhanced Financial Decision Making(Still Developing)

This repository implements FINCON, an LLM-based multi-agent framework with Conceptual Verbal Reinforcement for diverse financial tasks, as described in the paper "FINCON: A Synthesized LLM Multi-Agent System with Conceptual Verbal Reinforcement for Enhanced Financial Decision Making" (NeurIPS 2024).

## Overview

FINCON is a large language model (LLM) multi-agent system designed for financial decision-making tasks, including:
- Single stock trading
- Portfolio management

Inspired by real-world investment firm structures, FINCON employs a manager-analyst hierarchy with synchronized cross-functional agent collaboration, enabling more effective financial decision-making through verbal reinforcement mechanisms.

## Key Features

### Manager-Analyst Hierarchy
- **Manager Agent**: Consolidates insights and makes final trading decisions
- **Analyst Agents**:
  - **News Analyst**: Processes financial news articles
  - **Filing Analyst**: Analyzes SEC filings (10-K, 10-Q)
  - **ECC Analyst**: Extracts insights from earnings call conferences
  - **Data Analyst**: Computes technical indicators and market metrics
  - **Stock Selection Agent**: Performs portfolio construction and selection

### Dual-Level Risk Control
1. **Within-Episode Risk Control**: Monitors daily market risk using Conditional Value at Risk (CVaR)
2. **Over-Episode Risk Control**: Updates investment beliefs between episodes using Conceptual Verbal Reinforcement (CVRF)

### Conceptual Verbal Reinforcement
FINCON uses a novel CVRF mechanism that enables it to learn from experience by:
- Comparing performance between episodes
- Extracting conceptualized insights from both successful and unsuccessful trading patterns
- Updating investment beliefs through text-based gradient descent
- Selectively propagating updated beliefs to relevant agents

## Requirements

- Python 3.8+
- OpenAI API key (for GPT-4)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/username/fincon.git
   cd fincon
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```
   export OPENAI_API_KEY=your_openai_api_key
   ```
   Alternatively, create a `.env` file with these variables.

## Usage

### Running FINCON from the Command Line

```bash
python main.py --task_type single_stock_trading --symbols TSLA --max_episodes 4
```

#### Command-Line Arguments

- `--task_type`: Type of financial task (`single_stock_trading` or `portfolio_management`)
- `--symbols`: Stock symbols to trade (space-separated for multiple symbols)
- `--training_start`: Training start date (YYYY-MM-DD)
- `--training_end`: Training end date (YYYY-MM-DD)
- `--testing_start`: Testing start date (YYYY-MM-DD)
- `--testing_end`: Testing end date (YYYY-MM-DD)
- `--max_episodes`: Maximum number of training episodes
- `--llm_model`: LLM model name (defaults to "gpt-4-turbo")
- `--results_dir`: Directory to save results
- `--skip_visualizations`: Skip generating visualizations

### Example: Portfolio Management

```bash
python main.py --task_type portfolio_management --symbols TSLA MSFT AAPL --max_episodes 4
```

## Project Structure

```
fincon/
├── agents/
│   ├── manager_agent.py                # Manager agent implementation
│   ├── risk_control_agent.py           # Risk control agent implementation
│   ├── base_agent.py                   # Base agent class
│   └── analyst_agents/                 # Analyst agent implementations
│       ├── data_analyst.py             # Technical analysis agent
│       ├── news_analyst.py             # News processing agent
│       ├── filing_analyst.py           # SEC filing analysis agent
│       ├── ecc_analyst.py              # Earnings call analysis agent
│       └── stock_selection_agent.py    # Portfolio construction agent
├── memory/
│   ├── working_memory.py               # Working memory for short-term processing
│   ├── procedural_memory.py            # Procedural memory for historical actions
│   └── episodic_memory.py              # Episodic memory for manager agent
├── risk_control/
│   ├── within_episode.py               # Within-episode risk control
│   └── over_episode.py                 # Over-episode risk control with CVRF
├── evaluation/
│   ├── metrics.py                      # Performance metrics calculation
│   └── visualization.py                # Results visualization
├── utils/
│   ├── data_utils.py                   # Data loading and processing utilities
│   ├── financial_metrics.py            # Financial metrics calculation
│   ├── llm_utils.py                    # LLM client utilities
│   └── text_gradient_descent.py        # Text-based gradient descent for CVRF
├── modules/
│   ├── general_config.py               # General configuration module
│   ├── profiling.py                    # Agent profiling module
│   ├── perception.py                   # Perception module
│   └── action.py                       # Action module
├── data/                               # Data directory
│   ├── stock_data/                     # Stock price data
│   ├── news_data/                      # Financial news data
│   ├── filings/                        # SEC filings data
│   └── ecc_audio/                      # Earnings call data
├── results/                            # Results directory
├── config.py                           # System configuration
├── main.py                             # Main script
└── requirements.txt                    # Dependencies
```

## Results

After running FINCON, the results will be saved in the `results/` directory, including:
- Training and testing performance metrics
- Trading decisions and positions
- Visualizations (cumulative returns, trading positions, drawdowns, etc.)
- Summary report

## Configuration

The system configuration can be modified in `config.py`, including:
- LLM settings (model, temperature, API key)
- Training and testing periods
- Risk control parameters (CVaR confidence level, threshold)
- Initial investment amount
- Portfolio management settings

## Extending FINCON

FINCON can be extended in several ways:
1. Adding new analyst agent types for processing additional data sources
2. Implementing alternative risk control mechanisms
3. Supporting different financial markets and instruments
4. Enhancing verbal reinforcement mechanisms

## Citation

If you use FINCON in your research, please cite the original paper:

```
@inproceedings{yu2024fincon,
  title={FINCON: A Synthesized LLM Multi-Agent System with Conceptual Verbal Reinforcement for Enhanced Financial Decision Making},
  author={Yu, Yangyang and Yao, Zhiyuan and Li, Haohang and Deng, Zhiyang and Jiang, Yuechen and Cao, Yupeng and Chen, Zhi and Suchow, Jordan W. and Cui, Zhenyu and Liu, Rong and Xu, Zhaozhuo and Zhang, Denghui and Subbalakshmi, Koduvayur and Xiong, Guojun and He, Yueru and Huang, Jimin and Li, Dong and Xie, Qianqian},
  booktitle={38th Conference on Neural Information Processing Systems (NeurIPS 2024)},
  year={2024}
}
```

## License

MIT License