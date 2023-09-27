library(tidyverse)
library(pdftools)
library(jsonlite)

#' Het inlezen van data van de tweede kamer kan door middel van het OData-protocol.
#' Het CBS heeft hier een handleiding voor geschreven:
#' https://www.cbs.nl/nl-nl/onze-diensten/open-data/statline-als-open-data/snelstartgids


get_odata <- function(targetUrl) {
  targetUrl <- str_replace_all(targetUrl, " ", "%20")
  data <- data.frame()

  while(!is.null(targetUrl)){
    response <- fromJSON(url(targetUrl))
    data <- bind_rows(data,response$value)
    targetUrl <- response[["@odata.nextLink"]]
  }

  data <- data %>%
    mutate(url_document = paste0("https://gegevensmagazijn.tweedekamer.nl/OData/v4/2.0/Document(", Id, ")/resource"))
  return(data)
}


inlezen_pdf <- function(url){
  pdftools::pdf_text(url)
}


# filteren op documenten in 2022 van regering
url_docs_20_nu <- "https://gegevensmagazijn.tweedekamer.nl/OData/v4/2.0/Document?$filter=year(DatumOntvangst) ge 2020 and Soort eq 'Brief regering'"
docs_20_nu <- get_odata(url_docs_20_nu)
docs_20_nu_onderwijs <- docs_20_nu %>%
  filter(if_any(everything(), ~ str_detect(., "Onderwijs|onderwijs|Cultuur|cultuur|Wetenschap|wetenschap")))


test <- docs_20_nu_onderwijs %>%
  head(10)
test_txt <- test %>%
  mutate(tekst = map(url_document, inlezen_pdf)) %>%
  # tidyr::unnest_wider(tekst) %>%
  tidyr::unnest_longer(tekst) %>%
  dplyr::group_by(Id) %>%
  dplyr::mutate(pagina = row_number()) %>%
  dplyr::ungroup() %>%
  select(Id, pagina, everything())

