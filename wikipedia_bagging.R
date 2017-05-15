# Script to replicate bagging example from wikipedia
# The ozone data is from the ElemStatLearn package

library(ElemStatLearn)
library(plotly)

data(ozone)
x.pred <- seq.int(range(ozone$ozone)[1], range(ozone$ozone)[2], 1)
set.seed(257)

bag.wiki <- function(n = 100, interactive = TRUE) {
  p <- ggplot()

  for (i in 1:n) {
    row.idx <- sample(1:nrow(ozone), size = nrow(ozone), replace = TRUE)
    df <- ozone[row.idx, c(1,3)]
    m <- loess(temperature ~ ozone, data = df, family="gaussian", span=0.5, degree=1)
    pred <- predict(m, newdata = x.pred)
    gg.df <- data.frame(Temperature = x.pred, Ozone = pred)
    if (exists("overall")) {
      overall <- cbind(overall, pred)
    } else {
      overall <- pred 
    }
    p <- p + geom_line(data = gg.df, aes(x = Temperature, y = Ozone), na.rm = TRUE,
                       alpha = min(10/n, 0.2),  color = "black")
  }
  overall.means <- data.frame(Temperature = x.pred, Ozone = rowMeans(overall, na.rm = TRUE))
  
  p2 <- p + 
    geom_point(data = ozone, aes(x = ozone, y = temperature), 
                       shape = 21, color = "blue", stroke = 0.3) + 
    geom_line(data = overall.means, aes(x = Temperature, y = Ozone), na.rm = TRUE, 
              alpha = 1,  color = "red")
  
  if (interactive == TRUE) {
    ggplotly(p2)
  } else {
    p2
  }
  
}

bag.wiki(n = 200, interactive = TRUE)

