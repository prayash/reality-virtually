//
//  Datasource.swift
//  luminate
//
//  Created by Prayash Thapa on 10/7/17.
//  Copyright © 2017 com.reality.af. All rights reserved.
//

import LBTAComponents

class HomeDatasource: Datasource {
    
    override init() {
        super.init()
        objects = ["Forever Vegas Strong", "Sponsor a child at Sustainable Cambodia"]
    }
    
    override func cellClasses() -> [DatasourceCell.Type] {
        return [Card.self]
    }
    
    override func cellClass(_ indexPath: IndexPath) -> DatasourceCell.Type? {
        return Card.self
    }
}

