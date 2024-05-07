-- PathIE associated relations can now
UPDATE public.Predication SET relation = null WHERE extraction_type = 'PathIE' and relation = 'associated' ;
-- Ensure that all predication based on co-occurrences are mapped to associated
UPDATE public.Predication SET relation = 'associated' WHERE extraction_type = 'COSentence';