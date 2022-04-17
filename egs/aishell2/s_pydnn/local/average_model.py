#!/usr/bin/env python3
# -*- coding: utf-8 -*-    


if args.backend == "pytorch":
        import torch

        # sum
        for path in last:
            states = torch.load(path, map_location=torch.device("cpu"))["model"]
            if avg is None:
                avg = states
            else:
                for k in avg.keys():
                    avg[k] += states[k]

        # average
        for k in avg.keys():
            if avg[k] is not None:
                if avg[k].is_floating_point():
                    avg[k] /= args.num
                else:
                    avg[k] //= args.num

        torch.save(avg, args.out)
