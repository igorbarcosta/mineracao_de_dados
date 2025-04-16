


document.addEventListener('DOMContentLoaded', function () {
  // Gráfico de Barras: Número de estudantes por curso
  const ctxBarras = document.getElementById('graficoBarras');
  if (ctxBarras) {
    new Chart(ctxBarras, {
      type: 'bar',
      data: {
        labels: ['Engenharia', 'Direito', 'Psicologia', 'Administração'],
        datasets: [{
          label: 'Número de estudantes',
          data: [120, 90, 80, 100],
          backgroundColor: '#42a5f5'
        }]
      },
      options: {
        responsive: true,
        plugins: {
          legend: { display: false },
          title: {
            display: true,
            text: 'Número de estudantes por curso'
          }
        },
        scales: {
          y: {
            beginAtZero: true
          }
        }
      }
    });
  }

  // Gráfico de Pizza: Formas de pagamento preferidas
  const ctxPizza = document.getElementById('graficoPizza');
  if (ctxPizza) {
    new Chart(ctxPizza, {
      type: 'pie',
      data: {
        labels: ['Cartão', 'Dinheiro', 'Pix'],
        datasets: [{
          label: 'Formas de pagamento',
          data: [45, 30, 25],
          backgroundColor: ['#66bb6a', '#ffa726', '#29b6f6']
        }]
      },
      options: {
        responsive: true,
        plugins: {
          title: {
            display: true,
            text: 'Formas de pagamento preferidas'
          }
        }
      }
    });
  }

  // Histograma (simulado com gráfico de barras): Distribuição de alturas
  const ctxHistograma = document.getElementById('graficoHistograma');
  if (ctxHistograma) {
    new Chart(ctxHistograma, {
      type: 'bar',
      data: {
        labels: ['160-164', '165-169', '170-174', '175-179', '180-184', '185-189'],
        datasets: [{
          label: 'Frequência',
          data: [2, 4, 5, 3, 2, 1],
          backgroundColor: '#ab47bc'
        }]
      },
      options: {
        responsive: true,
        plugins: {
          legend: { display: false },
          title: {
            display: true,
            text: 'Distribuição de alturas (em cm)'
          }
        },
        scales: {
          y: {
            beginAtZero: true
          }
        }
      }
    });
  }

  // Gráfico de Linhas: Temperatura ao longo da semana
  const ctxLinha = document.getElementById('graficoLinha');
  if (ctxLinha) {
    new Chart(ctxLinha, {
      type: 'line',
      data: {
        labels: ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom'],
        datasets: [{
          label: 'Temperatura (°C)',
          data: [28, 29, 31, 30, 29, 27, 28],
          borderColor: '#ef5350',
          backgroundColor: '#ef9a9a',
          tension: 0.3,
          fill: true,
          pointRadius: 4
        }]
      },
      options: {
        responsive: true,
        plugins: {
          title: {
            display: true,
            text: 'Temperatura média diária'
          }
        },
        scales: {
          y: {
            beginAtZero: false
          }
        }
      }
    });
  }

  // Gráfico de Colunas Empilhadas: uso de transporte por faixa etária
const ctxEmpilhado = document.getElementById('graficoEmpilhado');
if (ctxEmpilhado) {
  new Chart(ctxEmpilhado, {
    type: 'bar',
    data: {
      labels: ['18-25', '26-35', '36-45', '46-60'],
      datasets: [
        {
          label: 'Ônibus',
          data: [40, 30, 20, 10],
          backgroundColor: '#42a5f5'
        },
        {
          label: 'Carro',
          data: [20, 35, 40, 30],
          backgroundColor: '#66bb6a'
        },
        {
          label: 'Bicicleta',
          data: [10, 5, 2, 1],
          backgroundColor: '#ffa726'
        }
      ]
    },
    options: {
      responsive: true,
      plugins: {
        title: {
          display: true,
          text: 'Meios de transporte por faixa etária'
        }
      },
      scales: {
        x: { stacked: true },
        y: {
          stacked: true,
          beginAtZero: true
        }
      }
    }
  });
}

// Gráfico de Boxplot: distribuição de salários (em R$)
const ctxBoxplot = document.getElementById('graficoBoxplot');
if (ctxBoxplot) {
  new Chart(ctxBoxplot, {
    type: 'boxplot',
    data: {
      labels: ['Salários'],
      datasets: [{
        label: 'Distribuição salarial',
        backgroundColor: '#ab47bc',
        borderColor: '#6a1b9a',
        borderWidth: 1,
        outlierColor: '#f44336',
        padding: 10,
        itemRadius: 0,
        data: [
          {
            min: 1800,
            q1: 2200,
            median: 2700,
            q3: 3200,
            max: 4000,
            outliers: [5000, 6000]
          }
        ]
      }]
    },
    options: {
      responsive: true,
      plugins: {
        title: {
          display: true,
          text: 'Distribuição de salários (em R$)'
        }
      }
    }
  });
}

// Gráfico de Dispersão: relação entre horas de estudo e nota
const ctxDispersao = document.getElementById('graficoDispersao');
if (ctxDispersao) {
  new Chart(ctxDispersao, {
    type: 'scatter',
    data: {
      datasets: [{
        label: 'Estudantes',
        backgroundColor: '#29b6f6',
        data: [
          { x: 1, y: 5 },
          { x: 2, y: 6 },
          { x: 3, y: 7 },
          { x: 4, y: 8 },
          { x: 5, y: 8.5 },
          { x: 6, y: 9 },
          { x: 7, y: 9.5 },
          { x: 8, y: 10 }
        ]
      }]
    },
    options: {
      responsive: true,
      plugins: {
        title: {
          display: true,
          text: 'Horas de estudo vs. nota final'
        }
      },
      scales: {
        x: {
          title: {
            display: true,
            text: 'Horas de estudo por semana'
          }
        },
        y: {
          title: {
            display: true,
            text: 'Nota final'
          }
        }
      }
    }
  });
}


});
