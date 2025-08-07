let portfolioChart = null;

// Initialize dashboard
document.addEventListener("DOMContentLoaded", function () {
  initPortfolioChart();
  fetchData();

  // Update every 30 seconds
  setInterval(fetchData, 30000);
});

function initPortfolioChart() {
  const ctx = document.getElementById("portfolio-chart").getContext("2d");
  portfolioChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Cash"],
      datasets: [
        {
          data: [1000],
          backgroundColor: ["#60a5fa"],
          borderWidth: 0,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: {
          position: "bottom",
          labels: {
            color: "#9ca3af",
            padding: 20,
          },
        },
      },
    },
  });
}

async function fetchData() {
  try {
    const response = await fetch("/api/status");
    const data = await response.json();

    updateSummaryCards(data);
    updateMarketData(data.market_data);
    updatePositions(data.positions, data.market_data);
    updateTradeHistory(data.trade_history);
    updatePortfolioChart(data);
  } catch (error) {
    console.error("Error fetching data:", error);
    // Show connection status
    document.querySelector(".text-green-400").innerHTML =
      '<i class="fas fa-circle text-xs mr-1 text-red-400"></i><span class="text-sm text-red-400">OFFLINE</span>';
  }
}

function updateSummaryCards(data) {
  document.getElementById(
    "total-value"
  ).textContent = `$${data.total_value.toFixed(2)}`;

  const pnlElement = document.getElementById("total-pnl");
  const pnl = data.total_value - 1000;
  pnlElement.textContent = `$${pnl >= 0 ? "+" : ""}${pnl.toFixed(2)}`;
  pnlElement.className = `text-2xl font-bold ${
    pnl >= 0 ? "text-green-400" : "text-red-400"
  }`;

  document.getElementById("active-positions").textContent = Object.keys(
    data.positions
  ).length;
  document.getElementById("total-trades").textContent =
    data.trade_history.length;
}

function updateMarketData(marketData) {
  const container = document.getElementById("market-data");
  container.innerHTML = "";

  Object.entries(marketData).forEach(([coin, data]) => {
    const changeColor =
      data.change_24h >= 0 ? "text-green-400" : "text-red-400";
    const changeIcon = data.change_24h >= 0 ? "fa-arrow-up" : "fa-arrow-down";

    container.innerHTML += `
            <div class="flex items-center justify-between p-4 bg-gray-700 rounded-lg">
                <div class="flex items-center">
                    <div class="w-10 h-10 bg-blue-500 rounded-full flex items-center justify-center mr-4">
                        <span class="font-bold text-sm">${coin}</span>
                    </div>
                    <div>
                        <p class="font-semibold">${coin}</p>
                        <p class="text-gray-400 text-sm">$${data.price.toLocaleString()}</p>
                    </div>
                </div>
                <div class="text-right">
                    <p class="${changeColor} font-semibold">
                        <i class="fas ${changeIcon} mr-1"></i>
                        ${
                          data.change_24h >= 0 ? "+" : ""
                        }${data.change_24h.toFixed(2)}%
                    </p>
                    <p class="text-gray-400 text-sm">24h</p>
                </div>
            </div>
        `;
  });
}

function updatePositions(positions, marketData) {
  const tbody = document.getElementById("positions-body");

  if (Object.keys(positions).length === 0) {
    tbody.innerHTML = `
            <tr>
                <td colspan="8" class="text-center py-8 text-gray-400">
                    <i class="fas fa-inbox text-4xl mb-4 block"></i>
                    No active positions
                </td>
            </tr>
        `;
    return;
  }

  tbody.innerHTML = "";

  Object.entries(positions).forEach(([coin, position]) => {
    const currentPrice = marketData[coin].price;
    const unrealizedPnl =
      (currentPrice - position.entry_price) * position.amount;
    const pnlPercent =
      (unrealizedPnl / (position.entry_price * position.amount)) * 100;
    const duration = calculateDuration(position.entry_time);

    const pnlColor = unrealizedPnl >= 0 ? "text-green-400" : "text-red-400";

    tbody.innerHTML += `
            <tr class="border-b border-gray-700">
                <td class="py-3 px-4 font-semibold">${coin}</td>
                <td class="py-3 px-4">${position.amount.toFixed(6)}</td>
                <td class="py-3 px-4">$${position.entry_price.toLocaleString()}</td>
                <td class="py-3 px-4">$${currentPrice.toLocaleString()}</td>
                <td class="py-3 px-4 ${pnlColor} font-semibold">
                    $${unrealizedPnl >= 0 ? "+" : ""}${unrealizedPnl.toFixed(2)}
                    <br><span class="text-sm">(${
                      pnlPercent >= 0 ? "+" : ""
                    }${pnlPercent.toFixed(1)}%)</span>
                </td>
                <td class="py-3 px-4">${duration}</td>
                <td class="py-3 px-4">$${position.stop_loss.toLocaleString()}</td>
                <td class="py-3 px-4">$${position.take_profit.toLocaleString()}</td>
            </tr>
        `;
  });
}

function updateTradeHistory(tradeHistory) {
  const tbody = document.getElementById("trades-body");

  if (tradeHistory.length === 0) {
    tbody.innerHTML = `
            <tr>
                <td colspan="7" class="text-center py-8 text-gray-400">
                    <i class="fas fa-chart-line text-4xl mb-4 block"></i>
                    No trades yet - AI is analyzing market conditions
                </td>
            </tr>
        `;
    return;
  }

  tbody.innerHTML = "";

  // Show last 10 trades
  tradeHistory
    .slice(-10)
    .reverse()
    .forEach((trade) => {
      const actionColor =
        trade.action === "BUY" ? "text-green-400" : "text-red-400";
      const actionIcon =
        trade.action === "BUY" ? "fa-arrow-up" : "fa-arrow-down";

      let pnlCell = "-";
      if (trade.profit_loss !== undefined) {
        const pnlColor =
          trade.profit_loss >= 0 ? "text-green-400" : "text-red-400";
        pnlCell = `<span class="${pnlColor}">$${
          trade.profit_loss >= 0 ? "+" : ""
        }${trade.profit_loss.toFixed(2)}</span>`;
      }

      tbody.innerHTML += `
            <tr class="border-b border-gray-700">
                <td class="py-3 px-4">${new Date(
                  trade.time
                ).toLocaleTimeString()}</td>
                <td class="py-3 px-4 font-semibold">${trade.coin}</td>
                <td class="py-3 px-4 ${actionColor}">
                    <i class="fas ${actionIcon} mr-1"></i>${trade.action}
                </td>
                <td class="py-3 px-4">$${trade.price.toLocaleString()}</td>
                <td class="py-3 px-4">${trade.amount.toFixed(6)}</td>
                <td class="py-3 px-4">$${trade.value.toFixed(2)}</td>
                <td class="py-3 px-4">${pnlCell}</td>
            </tr>
        `;
    });
}

function updatePortfolioChart(data) {
  const labels = ["Cash"];
  const values = [data.balance];
  const colors = ["#60a5fa"];

  Object.entries(data.positions).forEach(([coin, position]) => {
    const currentValue = position.amount * data.market_data[coin].price;
    labels.push(coin);
    values.push(currentValue);
    colors.push(getRandomColor());
  });

  portfolioChart.data.labels = labels;
  portfolioChart.data.datasets[0].data = values;
  portfolioChart.data.datasets[0].backgroundColor = colors;
  portfolioChart.update();
}

function calculateDuration(entryTime) {
  const now = new Date();
  const entry = new Date(entryTime);
  const diff = Math.floor((now - entry) / 1000);

  if (diff < 60) return `${diff}s`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h`;
  return `${Math.floor(diff / 86400)}d`;
}

function getRandomColor() {
  const colors = [
    "#10b981",
    "#f59e0b",
    "#ef4444",
    "#8b5cf6",
    "#06b6d4",
    "#84cc16",
  ];
  return colors[Math.floor(Math.random() * colors.length)];
}
