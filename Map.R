library(maps)
library(plyr)
library(ggplot2)
library(Hmisc)
library(tidyverse)
phy <- read.csv("rawdata.csv")
all_states <- map_data("state")
abb <- read_csv("region.csv")
all_states <- read_csv("allstate.csv")
colnames(phy)[colnames(phy)=="st"] <- "region"
rr <- subset(phy,select = c(25,38))
aa <- rr[which(rr$assgn=="M"),]
dd <- aa %>%
  group_by(region) %>%
  summarise(M=n())
Total <- left_join(all_states, dd, by="region")
p<- ggplot()+
  geom_polygon(data=Total, aes(x=long, y=lat, group=group,fill=Total$M), colour="black"
)+ scale_fill_continuous(low="lightsteelblue",high="midnightblue", guide="colorbar")
P1 <- p + theme_bw() +labs(fill="Number of Doctors",title= "Distribution of Non-Assignment MediCare Doctors by State", x="", y="")
P1 + scale_y_continuous(breaks=c()) + scale_x_continuous(breaks=c()) +  theme(panel.border= element_blank())

